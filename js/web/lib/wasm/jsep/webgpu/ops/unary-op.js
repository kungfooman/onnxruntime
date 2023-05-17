// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

//import {TensorView} from '../../tensor';
import {MAX_CLIP, MIN_CLIP, ShapeUtil} from '../../util.js';
import {/*AttributeWithCacheKey,*/ createAttributeWithCacheKey} from '../attribute-with-cache-key.js';
import {/*ComputeContext,*/ GpuDataType/*, ProgramInfo, ProgramInfoLoader, ProgramMetadata*/} from '../types.js';

//import {ShaderHelper} from './common.js';

//type BuiltinFunctionName = string;
//type ElementwiseCustomExpression = (expression: string) => string;
//type ElementwiseFunctionCall = BuiltinFunctionName|ElementwiseCustomExpression;
/**
 *
 * @param {ShaderHelper} shaderHelper
 * @param {number} datasize
 * @param {ElementwiseFunctionCall} funcCall
 * @param {string} [additionalImplementation]
 * @returns {string}
 */
const createElementwiseProgramShader = (
  shaderHelper, datasize, funcCall,
  additionalImplementation
) => {
      const vecSize = Math.ceil(datasize / 4);

      let expression = '';
      if (typeof funcCall === 'string') {
        expression = `${funcCall}(a)`;
      } else {
        expression = funcCall('a');
      }
      return `
  @group(0) @binding(0) var<storage, read> inputData : array<vec4<f32>>;
  @group(0) @binding(1) var<storage, read_write> outputData : array<vec4<f32>>;

  ${additionalImplementation ?? ''}

  ${shaderHelper.mainStart()}
    ${shaderHelper.guardAgainstOutOfBoundsWorkgroupSizes(vecSize)}

    let a = inputData[global_idx];
    outputData[global_idx] = ${expression};
  }`;
    };
/**
 *
 * @param {ProgramMetadata} metadata
 * @param {TensorView} input
 * @param {ElementwiseFunctionCall} funcCall
 * @param {string} [additionalImplementation]
 * @returns {ProgramInfo}
 */
const createElementwiseProgramInfo = (
  metadata, input, funcCall,
  additionalImplementation
) => ({
  ...metadata,
  getShaderSource: shaderHelper =>
      createElementwiseProgramShader(shaderHelper, ShapeUtil.size(input.dims), funcCall, additionalImplementation),
  outputs: [{dims: input.dims, dataType: input.dataType, gpuDataType: GpuDataType.default}],
  dispatchGroup: (inputTensors) =>
      ({x: Math.ceil(ShapeUtil.size(inputTensors[0].dims) / 64 /* workgroup size */ / 4 /* vec size */)})
});
/**
 *
 * @param {TensorView} input
 * @param {string} name
 * @param {ElementwiseFunctionCall} funcCall
 * @param {string} [additionalImplementation]
 * @param {string} [cacheKey]
 * @returns {ProgramInfoLoader}
 */
const createElementwiseProgramInfoLoader = (
  input, name, funcCall, additionalImplementation, cacheKey
) => {
  /** @type {ProgramMetadata} */
  const metadata = {name, inputTypes: [GpuDataType.default], cacheHint: cacheKey};
  return {
    ...metadata,
    get: () => createElementwiseProgramInfo(metadata, input, funcCall, additionalImplementation)
  };
};
/**
 * @param {ComputeContext} context
 * @returns {void}
 */
export const abs = (context) => {
  context.compute(createElementwiseProgramInfoLoader(context.inputs[0], 'Abs', 'abs'));
};
/**
 * @param {ComputeContext} context
 * @returns {void}
 */
export const acos = (context) => {
  context.compute(createElementwiseProgramInfoLoader(context.inputs[0], 'Acos', 'acos'));
};
/**
 * @param {ComputeContext} context
 * @returns {void}
 */
export const acosh = (context) => {
  context.compute(createElementwiseProgramInfoLoader(context.inputs[0], 'Acosh', 'acosh'));
};
/**
 * @param {ComputeContext} context
 * @returns {void}
 */
export const asin = (context) => {
  context.compute(createElementwiseProgramInfoLoader(context.inputs[0], 'Asin', 'asin'));
};
/**
 * @param {ComputeContext} context
 * @returns {void}
 */
export const asinh = (context) => {
  context.compute(createElementwiseProgramInfoLoader(context.inputs[0], 'Asinh', 'asinh'));
};
/**
 * @param {ComputeContext} context
 * @returns {void}
 */
export const atan = (context) => {
  context.compute(createElementwiseProgramInfoLoader(context.inputs[0], 'Atan', 'atan'));
};
/**
 * @param {ComputeContext} context
 * @returns {void}
 */
export const atanh = (context) => {
  context.compute(createElementwiseProgramInfoLoader(context.inputs[0], 'Atanh', 'atanh'));
};

//export interface ClipAttributes extends AttributeWithCacheKey {
//  readonly min: number;
//  readonly max: number;
//}
/**
 *
 * @param {ComputeContext} context
 * @param {ClipAttributes} attributes
 * @returns {void}
 */
export const clipV10 = (context, attributes) => {
  context.compute(
      createElementwiseProgramInfoLoader(
          context.inputs[0], 'Clip', a => `clamp(${a}, clip_min_, clip_max_)`, `
    const clip_min_: vec4<f32> = vec4(f32(${attributes.min}));
    const clip_max_: vec4<f32> = vec4(f32(${attributes.max}));
`,
          attributes.cacheKey),
      {inputs: [0]});
};
/**
 * @param {readonly TensorView[]} inputs
 * @returns {ClipAttributes}
 */
const generateClipAttributesFromInputs = (inputs) => {
  const min = (inputs.length >= 2) ? inputs[1].getFloat32Array()[0] : MIN_CLIP;
  const max = (inputs.length >= 3) ? inputs[2].getFloat32Array()[0] : MAX_CLIP;
  return createAttributeWithCacheKey({min, max});
};
/**
 *
 * @param {ComputeContext} context
 * @returns {void}
 */
export const clip = (context) => {
  const attributes = generateClipAttributesFromInputs(context.inputs);
  clipV10(context, attributes);
};
/**
 *
 * @param {ComputeContext} context
 * @returns {void}
 */
export const ceil = (context) => {
  context.compute(createElementwiseProgramInfoLoader(context.inputs[0], 'Ceil', 'ceil'));
};
/**
 * @param {ComputeContext} context
 * @returns {void}
 */
export const cos = (context) => {
  context.compute(createElementwiseProgramInfoLoader(context.inputs[0], 'Cos', 'cos'));
};
/**
 * @param {ComputeContext} context
 * @returns {void}
 */
export const cosh = (context) => {
  context.compute(createElementwiseProgramInfoLoader(context.inputs[0], 'Cosh', 'cosh'));
};

//export interface AlphaAttributes extends AttributeWithCacheKey {
//  readonly alpha: number;
//}
/**
 * @param {Record<string, unknown>} attributes
 * @returns {AlphaAttributes}
 */
export const parseAlphaAttributes = (attributes) =>
    createAttributeWithCacheKey(attributes /*as {alpha: number}*/);
/**
 *
 * @param {ComputeContext} context
 * @param {AlphaAttributes} attributes
 * @returns {void}
 */
export const elu = (context, attributes) => {
  context.compute(createElementwiseProgramInfoLoader(
      context.inputs[0], 'Elu', a => `elu_vf32(${a})`, `
  const elu_alpha_: f32 = f32(${attributes.alpha});

  fn elu_f32(a: f32) -> f32 {
  return select((exp(a) - 1.0) * elu_alpha_, a, a >= 0.0);
  }

  fn elu_vf32(v: vec4<f32>) -> vec4<f32> {
  return vec4(elu_f32(v.x), elu_f32(v.y), elu_f32(v.z), elu_f32(v.w));
  }`,
      attributes.cacheKey));
};
/**
 * @param {ComputeContext} context
 * @returns {void}
 */
export const erf = (context) => {
  context.compute(createElementwiseProgramInfoLoader(context.inputs[0], 'Erf', a => `erf_vf32(${a})`, `
  const r0: f32 = 0.3275911;
  const r1: f32 = 0.254829592;
  const r2: f32 = -0.284496736;
  const r3: f32 = 1.421413741;
  const r4: f32 = -1.453152027;
  const r5: f32 = 1.061405429;

  fn erf_vf32(v: vec4<f32>) -> vec4<f32> {
    let absv = abs(v);
    let x = 1.0 / (1.0 + r0 * absv);
    return sign(v) * (1.0 - ((((r5 * x + r4) * x + r3) * x + r2) * x + r1) * x * exp(-absv * absv));
  }`));
};
/**
 * @param {ComputeContext} context
 * @returns {void}
 */
export const exp = (context) => {
  context.compute(createElementwiseProgramInfoLoader(context.inputs[0], 'Exp', 'exp'));
};
/**
 * @param {ComputeContext} context
 * @returns {void}
 */
export const floor = (context) => {
  context.compute(createElementwiseProgramInfoLoader(context.inputs[0], 'Floor', 'floor'));
};
/**
 *
 * @param {ComputeContext} context
 * @param {AlphaAttributes} attributes
 * @returns {void}
 */
export const leakyRelu = (context, attributes) => {
  context.compute(createElementwiseProgramInfoLoader(
      context.inputs[0], 'LeakyRelu', a => `select(leaky_relu_alpha_ * ${a}, ${a}, ${a} >= vec4<f32>(0.0))`,
      `const leaky_relu_alpha_: f32 = f32(${attributes.alpha});`, attributes.cacheKey));
};
/**
 * @param {ComputeContext} context
 * @returns {void}
 */
export const neg = (context) => {
  context.compute(createElementwiseProgramInfoLoader(context.inputs[0], 'Neg', a => `-${a}`));
};
/**
 * @param {ComputeContext} context
 * @returns {void}
 */
export const reciprocal = (context) => {
  context.compute(createElementwiseProgramInfoLoader(context.inputs[0], 'Reciprocal', a => `1.0/${a}`));
};
/**
 * @param {ComputeContext} context
 * @returns {void}
 */
export const relu = (context) => {
  context.compute(createElementwiseProgramInfoLoader(
      context.inputs[0], 'Relu', a => `select(vec4<f32>(0.0), ${a}, ${a} > vec4<f32>(0.0))`));
};
/**
 * @param {ComputeContext} context
 * @returns {void}
 */
export const sigmoid = (context) => {
  context.compute(createElementwiseProgramInfoLoader(context.inputs[0], 'Sigmoid', a => `(1.0 / (1.0 + exp(-${a})))`));
};
/**
 * @param {ComputeContext} context
 * @returns {void}
 */
export const sin = (context) => {
  context.compute(createElementwiseProgramInfoLoader(context.inputs[0], 'Sin', 'sin'));
};
/**
 * @param {ComputeContext} context
 * @returns {void}
 */
export const sinh = (context) => {
  context.compute(createElementwiseProgramInfoLoader(context.inputs[0], 'Sinh', 'sinh'));
};
/**
 * @param {ComputeContext} context
 * @returns {void}
 */
export const sqrt = (context) => {
  context.compute(createElementwiseProgramInfoLoader(context.inputs[0], 'Sqrt', 'sqrt'));
};
/**
 * @param {ComputeContext} context
 * @returns {void}
 */
export const tan = (context) => {
  context.compute(createElementwiseProgramInfoLoader(context.inputs[0], 'Tan', 'tan'));
};
/**
 * @param {ComputeContext} context
 * @returns {void}
 */
export const tanh = (context) => {
  context.compute(createElementwiseProgramInfoLoader(context.inputs[0], 'Tanh', 'tanh'));
};
/**
 * @param {ComputeContext} context
 * @param {AlphaAttributes} attributes
 * @returns {number}
 */
export const thresholdedRelu = (context, attributes) => {
  context.compute(createElementwiseProgramInfoLoader(
      context.inputs[0], 'ThresholdedRelu', a => `select(vec4<f32>(0.0), ${a}, ${a} > thresholded_relu_alpha_)`,
      `const thresholded_relu_alpha_: vec4<f32> = vec4<f32>(${attributes.alpha});`, attributes.cacheKey));
  return 0;
};
