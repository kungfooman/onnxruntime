// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

//import {Env} from 'onnxruntime-common';

//import {OrtWasmModule} from '../binding/ort-wasm.js';
import {getTensorElementSize} from '../wasm-common.js';

import {WebGpuBackend} from './backend-webgpu.js';
import {LOG_DEBUG} from './log.js';
//import {TensorView} from './tensor.js';
import {ShapeUtil} from './util.js';
import {
  //ComputeContext,
  //ComputeContextInputsOutputsMapping,
  //ProgramInfo,
  //ProgramInfoLoader
} from './webgpu/types.js';

/* eslint-disable no-bitwise */
/**
 * @implements {TensorView}
 */
class TensorViewImpl {
  /**
   * @type {OrtWasmModule}
   * @private
   */
  module;
  /**
   * @readonly
   * @type {number}
   */
  dataType;
  /**
   * @readonly
   * @type {number}
   */
  data;
  /**
   * @readonly
   * @type {readonly number[]}
   */
  dims;
  /**
   *
   * @param {OrtWasmModule} module
   * @param {number} dataType
   * @param {*} data
   * @param {*} dims
   */
  constructor(module, dataType, data, dims) {
    this.module = module;
    this.dataType = dataType;
    this.data = data;
    this.dims = dims;
  }
  /**
   * @returns {Float32Array}
   */
  getFloat32Array() {
    return new Float32Array(this.module.HEAP8.buffer, this.data, ShapeUtil.size(this.dims));
  }
  /**
   * @param {readonly number[]} newDims
   * @returns {TensorView}
   */
  reshape(newDims) {
    if (ShapeUtil.size(newDims) !== ShapeUtil.size(this.dims)) {
      throw new Error('Invalid new shape');
    }
    return new TensorViewImpl(this.module, this.dataType, this.data, newDims);
  }
}

/**
 * @implements {ComputeContext}
 */
class ComputeContextImpl {
  /**
   * @readonly
   * @type {number}
   */
  opKernelContext;
  /**
   * @readonly
   * @type {readonly TensorView[]}
   */
  inputs;
  /**
   * @type {{[key: string]: unknown}}
   */
  get customData() {
    return this.backend.currentKernelCustomData;
  }
  /**
   * @private
   * @type {OrtWasmModule}
   */
  module;
  /**
   * @private
   * @type {WebGpuBackend}
   */
  backend;
  /**
   *
   * @param {OrtWasmModule} module
   * @param {WebGpuBackend} backend
   * @param {number} contextDataOffset
   */
  constructor(module, backend, contextDataOffset) {
    this.module = module;
    this.backend = backend;
    const heapU32 = module.HEAPU32;

    // extract context data
    let dataIndex = (contextDataOffset >> 2);
    this.opKernelContext = heapU32[dataIndex++];
    const inputCount = heapU32[dataIndex++];
    /** @type {TensorView[]} */
    const inputs = [];
    for (let i = 0; i < inputCount; i++) {
      const dataType = heapU32[dataIndex++];
      const data = heapU32[dataIndex++];
      const dim = heapU32[dataIndex++];
      /** @type {number[]} */
      const dims = [];
      for (let d = 0; d < dim; d++) {
        dims.push(heapU32[dataIndex++]);
      }
      inputs.push(new TensorViewImpl(module, dataType, data, dims));
    }
    this.inputs = inputs;
  }
  /**
   *
   * @param {ProgramInfoLoader|ProgramInfo} program
   * @param {ComputeContextInputsOutputsMapping} [inputsOutputsMapping]
   * @returns {TensorView[]}
   */
  compute(program, inputsOutputsMapping) {
    // prepare inputs. inputs should always be valid data.
    const mappedInputs =
        inputsOutputsMapping?.inputs?.map(i => typeof i === 'number' ? this.inputs[i] : i) ?? this.inputs;
    // prepare outputs.
    const outputIndices = inputsOutputsMapping?.outputs ?? [];
    /**
     *
     * @param {number} index
     * @param {number} dataType
     * @param {readonly number[]} dims
     * @returns {TensorView}
     */
    const createKernelOutput = (index, dataType, dims) =>
        new TensorViewImpl(this.module, dataType, this.output(index, dims), dims);
    /**
     *
     * @param {number} dataType
     * @param {readonly number[]} dims
     * @returns {TensorView}
     */
    const createTemporaryOutput = (dataType, dims) => {
      const elementSize = getTensorElementSize(dataType);
      if (!elementSize) {
        throw new Error(`Unsupported data type: ${dataType}`);
      }
      const bufferSize = elementSize * ShapeUtil.size(dims);
      return new TensorViewImpl(this.module, dataType, this.backend.gpuDataManager.create(bufferSize).id, dims);
    };
    return this.backend.run(program, mappedInputs, outputIndices, createKernelOutput, createTemporaryOutput);
  }
  /**
   *
   * @param {number} index
   * @param {readonly number[]} dims
   * @returns {number}
   */
  output(index, dims) {
    const stack = this.module.stackSave();
    try {
      const data = this.module.stackAlloc((1 + dims.length) * 4 /* sizeof(size_t) */);
      let offset = data >> 2;
      this.module.HEAPU32[offset++] = dims.length;
      for (let i = 0; i < dims.length; i++) {
        this.module.HEAPU32[offset++] = dims[i];
      }
      return this.module._JsepOutput(this.opKernelContext, index, data);
    } finally {
      this.module.stackRestore(stack);
    }
  }
}
/**
 *
 * @param {OrtWasmModule} module
 * @param {Env} env
 * @returns {Promise<void>}
 */
export const init = async(module, env) => {
  const init = module.jsepInit;
  if (init && navigator.gpu) {
    if (!env.wasm.simd) {
      throw new Error(
          'Not supported for WebGPU=ON and SIMD=OFF. Please set `env.wasm.simd` to true when using WebGPU EP');
    }
    const backend = new WebGpuBackend();
    await backend.initialize(env);

    init(
        // backend
        {backend},
        /**
         * jsepAlloc()
         * @param {number} size
         * @returns {number}
         */
        (size) => backend.alloc(size),
        /**
         * jsepFree()
         * @param {number} ptr
         * @returns {number}
         */
        (ptr) => backend.free(ptr),
        /**
         * jsepCopy(src, dst, size, isSourceGpu)
         * @param {number} src
         * @param {number} dst
         * @param {number} size
         * @param {boolean} isSourceGpu
         */
        (src/*: */, dst/*: */, size/*: */, isSourceGpu = false) => {
          if (isSourceGpu) {
            LOG_DEBUG('verbose', () => `[WebGPU] jsepCopyGpuToGpu: src=${src}, dst=${dst}, size=${size}`);
            backend.memcpy(src, dst);
          } else {
            LOG_DEBUG('verbose', () => `[WebGPU] jsepCopyCpuToGpu: dataOffset=${src}, gpuDataId=${dst}, size=${size}`);
            const data = module.HEAPU8.subarray(src, src + size);
            backend.upload(dst, data);
          }
        },
        /**
         * jsepCopyAsync(src, dst, size)
         * @param {number} gpuDataId
         * @param {number} dataOffset
         * @param {number} size
         * @returns {Promise<void>}
         */
        async(gpuDataId, dataOffset, size) => {
          LOG_DEBUG(
            'verbose',
            () => `[WebGPU] jsepCopyGpuToCpu: gpuDataId=${gpuDataId}, dataOffset=${dataOffset}, size=${size}`
          );
          await backend.download(gpuDataId, () => module.HEAPU8.subarray(dataOffset, dataOffset + size));
        },
        /**
         * jsepCreateKernel
         * @param {string} name
         * @param {number} kernel
         * @param {unknown} attribute
         */
        (name, kernel, attribute) => backend.createKernel(name, kernel, attribute),
        /**
         * jsepReleaseKernel
         * @param {number} kernel
         * @returns {void}
         */
        (kernel) => backend.releaseKernel(kernel),
        /**
         * jsepRun
         * @param {number} kernel
         * @param {number} contextDataOffset
         * @returns {number}
         */
        (kernel, contextDataOffset) => {
          LOG_DEBUG('verbose', () => `[WebGPU] jsepRun: kernel=${kernel}, contextDataOffset=${contextDataOffset}`);
          const context = new ComputeContextImpl(module, backend, contextDataOffset);
          return backend.computeKernel(kernel, context);
        });
  }
};
