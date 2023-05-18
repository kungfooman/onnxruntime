// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

//import {Env} from 'onnxruntime-common';

import {configureLogger, LOG_DEBUG} from './log.js';
//import {TensorView} from './tensor.js';
import {createGpuDataManager, GpuDataManager} from './webgpu/gpu-data-manager.js';
import {
  // RunFunction,
  WEBGPU_OP_RESOLVE_RULES
} from './webgpu/op-resolve-rules.js';
import {ProgramManager} from './webgpu/program-manager.js';
import {
  //ComputeContext,
  //GpuData,
  GpuDataType,
  //ProgramInfo,
  //ProgramInfoLoader
} from './webgpu/types.js';

/**
 * get a unique key representing the program from the program info,input shapes and types.
 * @param {ProgramInfo|ProgramInfoLoader} programInfo
 * @param {ReadonlyArray<TensorView['dims']>} inputTensorShapes
 * @param {readonly GpuDataType[]} inputGpuDataTypes
 * @returns {string} a unique key is a shorter string than the shader source, which contains all the information to identify a
 * program. if the key is the same, the program shader source should be the same, so we can reuse the program.
 *
 */
const getProgramInfoUniqueKey = (programInfo, inputTensorShapes, inputGpuDataTypes) => {
      const inputTensorShapesToString = inputTensorShapes.map(d => `${d.join(',')}`).join('_');
      const inputGpuDataTypesToString = inputGpuDataTypes.join('_');
      let key = programInfo.name;
      if (programInfo.cacheHint) {
        key += '[' + programInfo.cacheHint + ']';
      }
      key += ':' + inputTensorShapesToString + ';' + inputGpuDataTypesToString;
      return key;
    };

/**
 * this class is designed to store status and being used as a singleton for JSEP. It will be passed to jsepInit() as
 * the first parameter so that it is stored for future use.
 */
export class WebGpuBackend {
  /**
   * @type {GPUDevice}
   */
  device;
  /**
   * an instance of GpuDataManager to manage a GpuDataId -> GpuBuffer mapping
   * @type {GpuDataManager}
   */
  gpuDataManager;
  /**
   * an instance of ProgramManager to build and run WebGPU compute shader program, and manage a ProgramKey -> Program
   * artifacts mapping
   * @type {ProgramManager}
   */
  programManager;

  /**
   * representing the kernel ID of which is currently being computed (CPU code perspective).
   * `null` means no kernel is being computed.
   * only one kernel can be computed at a moment.
   * @type {number|null}
   */
  currentKernelId = null;
  /**
   * a list of temporary GPU data for the current kernel. should release when the kernel done computation.
   * @type {GpuData[]}
   * @private
   */
  temporaryData;
  /**
   * a KernelID -> a GPU data list, which stores persistent GPU data owned by the specific kernel.
   * @type {Map<number, GpuData[]>}
   * @private
   */
  kernelPersistentData;
  /**
   * a KernelID -> a custom data, which stores custom data owned by the specific kernel.
   * @type {Map<number, {[key: string]: unknown}>}
   * @private
   */
  kernelCustomData;
  /**
   * get the custom data of the current kernel
   * @type {{[key: string]: unknown}}
   */
  get currentKernelCustomData() {
    if (this.currentKernelId === null) {
      throw new Error('currentKernelCustomData(): currentKernelId is null. (should not happen)');
    }

    let data = this.kernelCustomData.get(this.currentKernelId);
    if (!data) {
      data = {};
      this.kernelCustomData.set(this.currentKernelId, data);
    }

    return data;
  }

  /**
   * a KernelID -> kernel info mapping. value is [ name, run function, [optional] preprocess_attribute_once function ]
   * @type {Map<number, [string, RunFunction, [((attribute: unknown) => unknown) | undefined, unknown]]>}
   */
  kernels;
  /** @type {GPUCommandEncoder|null} */
  commandEncoder = null;
  /** @type {GPUComputePassEncoder|null} */
  computePassEncoder = null;
  pendingDispatchNumber = 0;

  profilingEnabled = false;
  /** @type {GPUQuerySet} */
  profilingQuerySet;
  /** @type {bigint|undefined} */
  profilingTimeBase;
  /**
   *
   * @param {Env} env
   * @returns {Promise<void>}
   */
  async initialize(env) {
    if (!navigator.gpu) {
      // WebGPU is not available.
      throw new Error('WebGpuBackend: WebGPU is not available.');
    }

    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) {
      throw new Error('WebGpuBackend: Failed to get GPU adapter.');
    }
    console.log("adapter.limits.maxComputeWorkgroupStorageSize", adapter.limits);
    /** @type {GPUDeviceDescriptor} */
    const deviceDescriptor = {
      requiredLimits: {
        // This crashes in Firefox Nightly
        //maxComputeWorkgroupStorageSize: adapter.limits.maxComputeWorkgroupStorageSize,
        maxComputeWorkgroupsPerDimension: adapter.limits.maxComputeWorkgroupsPerDimension,
        maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
      }
    };
    console.log({deviceDescriptor});
    // WebGPU Spec: Timestamp Queries Inside Passes
    // https://github.com/gpuweb/gpuweb/blob/main/proposals/timestamp-query-inside-passes.md
    if (adapter.features.has('timestamp-query-inside-passes') && env.webgpu.profilingMode === 'default') {
      this.profilingEnabled = true;
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      deviceDescriptor.requiredFeatures = ['timestamp-query-inside-passes' /*as any*/];
    }

    this.device = await adapter.requestDevice(deviceDescriptor);
    this.gpuDataManager = createGpuDataManager(this);
    this.programManager = new ProgramManager(this);
    this.kernels = new Map();
    this.kernelPersistentData = new Map();
    this.kernelCustomData = new Map();

    // set up flags for logger
    configureLogger(env.logLevel/*!*/, !!env.debug);

    // TODO: set up flags

    this.device.onuncapturederror = ev => {
      if (ev.error instanceof GPUValidationError) {
        // eslint-disable-next-line no-console
        console.error(`An uncaught WebGPU validation error was raised: ${ev.error.message}`);
      }
    };

    if (this.profilingEnabled) {
      this.profilingQuerySet = this.device.createQuerySet({
        type: 'timestamp',
        count: 2,
      });
    }
  }
  /**
   * @returns {void}
   */
  dispose() {
    // currently, we do not do anything in this function. In all known use cases, we don't have the requirement to
    // actually dispose the WebGpuBackend instance, because it's always used as a singleton.
    //
    // revisit this place if we get real requirement to dispose the instance.
  }
  /**
   *
   * @returns {GPUCommandEncoder}
   */
  getCommandEncoder() {
    if (!this.commandEncoder) {
      this.commandEncoder = this.device.createCommandEncoder();
    }
    return this.commandEncoder;
  }
  /**
   *
   * @returns {GPUComputePassEncoder}
   */
  getComputePassEncoder() {
    if (!this.computePassEncoder) {
      this.computePassEncoder = this.getCommandEncoder().beginComputePass();
    }
    return this.computePassEncoder;
  }
  /**
   * @returns {void}
   */
  endComputePass() {
    if (this.computePassEncoder) {
      this.computePassEncoder.end();
      this.computePassEncoder = null;
    }
  }
  /**
   * @returns {void}
   */
  flush() {
    this.endComputePass();
    this.device.queue.submit([this.getCommandEncoder().finish()]);
    this.gpuDataManager.refreshPendingBuffers();
    this.commandEncoder = null;
    this.pendingDispatchNumber = 0;
  }

  /**
   * run a WebGPU program.
   * @param {ProgramInfoLoader|ProgramInfo} program either a ProgramInfo instance containing metadata including the shader code, or a function that
   * can be called and return a ProgramInfo instance
   * @param {readonly TensorView[]} inputs a TensorView array. each element represents a value already exists in GPU.
   * @param {readonly number[]} outputIndices an indices array. each element can be either -1 (temporary data), -2 (persistent data) or an
   * index to the kernel's output.
   * @param {(index: number, dataType: number, dims: readonly number[]) => TensorView} createKernelOutput a callback function that create a value to kernel's output with the given index
   * @param {(dataType: number, dims: readonly number[]) => TensorView} createIntermediateOutput a callback function that create a value as a intermediate value, either temporary
   * or persistent (owned by the current kernel)
   * @returns {TensorView[]} a TensorView array representing the result.
   */
  run(
    program, inputs, outputIndices, createKernelOutput, createIntermediateOutput
  ) {
    if (inputs.length !== program.inputTypes.length) {
      throw new Error(`Input size must be equal to ${program.inputTypes.length}.`);
    }

    // create info for inputs
    /** @type {GpuData[]} */
    const inputDatas = [];
    for (let i = 0; i < inputs.length; ++i) {
      const gpuData = this.gpuDataManager.get(inputs[i].data);
      if (!gpuData) {
        throw new Error(`no GPU data for input: ${inputs[i].data}`);
      }
      inputDatas[i] = gpuData;
    }

    const key = getProgramInfoUniqueKey(program, inputs.map(i => i.dims), inputDatas.map(i => i.type));
    let artifact = this.programManager.getArtifact(key);
    const programInfo = artifact ?
        artifact.programInfo :
        (typeof (program /*as ProgramInfoLoader*/).get === 'function' ? (program /*as ProgramInfoLoader*/).get() :
                                                                    (program /*as ProgramInfo*/));

    // check output indices
    const validatedOutputIndices = outputIndices.length === 0 ? programInfo.outputs.map((_, i) => i) : outputIndices;
    if (validatedOutputIndices.length !== programInfo.outputs.length) {
      throw new Error(`Output size ${validatedOutputIndices.length} must be equal to ${programInfo.outputs.length}.`);
    }

    // create info for outputs
    /** @type {TensorView[]} */
    const outputTensorViews = [];
    /** @type {GpuData[]} */
    const outputDatas = [];
    for (let i = 0; i < programInfo.outputs.length; ++i) {
      // value -1 and -2 are used for creating temporary and persistent outputs. so -2, -1 and 0, 1, 2, ... are valid
      // output indices. see type definition of ComputeContextInputsOutputsMapping for more details.
      if (!Number.isInteger(validatedOutputIndices[i]) || validatedOutputIndices[i] < -2 ||
          validatedOutputIndices[i] >= programInfo.outputs.length) {
        throw new Error(`Invalid output index: ${validatedOutputIndices[i]}`);
      }
      const isTemporary = validatedOutputIndices[i] === -1;
      const isPersistent = validatedOutputIndices[i] === -2;
      const tensorView = (isTemporary || isPersistent) ?
          createIntermediateOutput(programInfo.outputs[i].dataType, programInfo.outputs[i].dims) :
          createKernelOutput(validatedOutputIndices[i], programInfo.outputs[i].dataType, programInfo.outputs[i].dims);
      const gpuData = this.gpuDataManager.get(tensorView.data);
      if (!gpuData) {
        throw new Error(`no GPU data for output: ${tensorView.data}`);
      }
      if (isTemporary) {
        this.temporaryData.push(gpuData);
      }
      if (isPersistent) {
        let persistentData = this.kernelPersistentData.get(this.currentKernelId/*!*/);
        if (!persistentData) {
          persistentData = [];
          this.kernelPersistentData.set(this.currentKernelId/*!*/, persistentData);
        }
        persistentData.push(gpuData);
      }
      outputTensorViews.push(tensorView);
      outputDatas.push(gpuData);
    }

    const normalizedDispatchGroup = this.programManager.normalizeDispatchGroupSize(programInfo.dispatchGroup(inputs));

    if (!artifact) {
      artifact = this.programManager.build(programInfo, normalizedDispatchGroup);
      this.programManager.setArtifact(key, artifact);
    }

    LOG_DEBUG(
        'info',
        () => `[ProgramManager] run "${programInfo.name}" (key=${key}) with ${normalizedDispatchGroup[0]}x${
            normalizedDispatchGroup[1]}x${normalizedDispatchGroup[2]}`);
    this.programManager.run(artifact, inputDatas, outputDatas, normalizedDispatchGroup);

    return outputTensorViews;
  }
  /**
   *
   * @param {number} gpuDataId
   * @param {Uint8Array} data
   * @returns {void}
   */
  upload(gpuDataId, data) {
    this.gpuDataManager.upload(gpuDataId, data);
  }
  /**
   *
   * @param {number} src
   * @param {number} dst
   * @returns {void}
   */
  memcpy(src, dst) {
    this.gpuDataManager.memcpy(src, dst);
  }
  /**
   *
   * @param {number} gpuDataId
   * @param {() => Uint8Array} getTargetBuffer
   * @returns {Promise<void>}
   */
  async download(gpuDataId, getTargetBuffer) {
    const arrayBuffer = await this.gpuDataManager.download(gpuDataId);

    // the underlying buffer may be changed after the async function is called. so we use a getter function to make sure
    // the buffer is up-to-date.
    const data = getTargetBuffer();
    data.set(new Uint8Array(arrayBuffer));
  }
  /**
   *
   * @param {number} size
   * @returns {number}
   */
  alloc(size) {
    return this.gpuDataManager.create(size).id;
  }
  /**
   *
   * @param {number} ptr
   * @returns {number}
   */
  free(ptr) {
    return this.gpuDataManager.release(ptr);
  }
  /**
   *
   * @param {string} name
   * @param {number} kernelId
   * @param {unknown} attribute
   * @returns {void}
   */
  createKernel(name, kernelId, attribute) {
    const op = WEBGPU_OP_RESOLVE_RULES.get(name);
    if (!op) {
      throw new Error(`kernel not implemented: ${name}`);
    }

    this.kernels.set(kernelId, [name, op[0], [op[1], attribute]]);
  }
  /**
   *
   * @param {number} kernelId
   * @returns {void}
   */
  releaseKernel(kernelId) {
    const persistentData = this.kernelPersistentData.get(kernelId);
    if (persistentData) {
      for (const data of persistentData) {
        this.gpuDataManager.release(data.id);
      }
      this.kernelPersistentData.delete(kernelId);
    }

    this.kernelCustomData.delete(kernelId);
    this.kernels.delete(kernelId);
  }
  /**
   *
   * @param {number} kernelId
   * @param {ComputeContext} context
   * @returns {number}
   */
  computeKernel(kernelId, context) {
    const kernel = this.kernels.get(kernelId);
    if (!kernel) {
      throw new Error(`kernel not created: ${kernelId}`);
    }
    const [name, kernelEntry, attributes] = kernel;
    if (this.currentKernelId !== null) {
      throw new Error(`kernel "${name}" is not allowed to be called recursively`);
    }
    this.currentKernelId = kernelId;

    // parse attributes if necessary
    if (attributes[0]) {
      attributes[1] = attributes[0](attributes[1]);
      attributes[0] = undefined;
    }

    LOG_DEBUG('info', () => `[WebGPU] Start to run kernel "${name}"...`);

    this.temporaryData = [];
    try {
      kernelEntry(context, attributes[1]);
      return 0;  // ORT_OK
    } catch (e) {
      LOG_DEBUG('warning', `[WebGPU] Kernel "${name}" failed. Error: ${e}`);
      return 1;  // ORT_FAIL
    } finally {
      for (const data of this.temporaryData) {
        this.gpuDataManager.release(data.id);
      }
      this.temporaryData = [];
      this.currentKernelId = null;
    }
  }
}
