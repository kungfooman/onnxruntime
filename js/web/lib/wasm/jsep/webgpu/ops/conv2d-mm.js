// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

//import {TensorView} from '../../tensor.js';
import {GpuDataType, /*ProgramInfoLoader, ProgramMetadata*/} from '../types.js';

import {createConv2DMatMulProgramInfo} from './3rd-party/conv2d_mm_webgpu.js';
//import {ConvAttributes} from './conv.js';

/**
 *
 * @param {boolean} hasBias
 * @param {string} cacheHint
 * @returns {ProgramMetadata}
 */
const createConv2DMatMulProgramMetadata = (hasBias, cacheHint) => ({
  name: 'Conv2DMatMul',
  inputTypes: hasBias ? [GpuDataType.default, GpuDataType.default, GpuDataType.default] :
                        [GpuDataType.default, GpuDataType.default],
  cacheHint
});
/**
 * @param {readonly TensorView[]} inputs
 * @param {ConvAttributes} attributes
 * @param {readonly number[]} outputShape
 * @param {number} dimAOuter
 * @param {number} dimBOuter
 * @param {number} dimInner
 * @param {boolean} hasBias
 * @param {boolean} sequentialAccessByThreads
 * @returns {ProgramInfoLoader}
 */
export const createConv2DMatMulProgramInfoLoader = (
  inputs, attributes, outputShape, dimAOuter,
  dimBOuter, dimInner, hasBias, sequentialAccessByThreads
) => {
  const metadata = createConv2DMatMulProgramMetadata(hasBias, attributes.cacheKey);
  return {
    ...metadata,
    get: () => createConv2DMatMulProgramInfo(
        inputs, metadata, attributes, outputShape, dimAOuter, dimBOuter, dimInner, hasBias,
        sequentialAccessByThreads)
  };
};
