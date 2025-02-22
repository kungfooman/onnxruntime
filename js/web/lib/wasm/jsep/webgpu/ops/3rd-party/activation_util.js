/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

// sampled from [@tensorflow/tfjs] tfjs-backend-webgpu/src/activation_util.ts
//
// modified to fit the needs of the project

//export declare type Activation = 'linear' | 'relu' | 'prelu' | 'elu' | 'relu6' | 'leakyrelu' | 'sigmoid';
/**
 * @param {number} component
 * @throws {Error}
 */
export const typeSnippet = (component) => {
  switch (component) {
    case 1:
      return 'f32';
    case 2:
      return 'vec2<f32>';
    case 3:
      return 'vec3<f32>';
    case 4:
      return 'vec4<f32>';
    default:
      throw new Error(`${component}-component is not supported.`);
  }
};
/**
 *
 * @param {Activation} [activation]
 * @param {boolean} [_hasPreluActivationWeights]
 * @param {boolean} [_packed]
 * @param {number} [_coordsLength]
 * @returns {string}
 */
export const activationFnSnippet =
    (activation, _hasPreluActivationWeights = false, _packed = false, _coordsLength = 3) => {
      if (!activation) {
        return '';
      }

      // TODO: add implementations
      return '';
    };
/**
 *
 * @param {boolean} hasBias
 * @param {Activation} [activation]
 * @returns {string}
 */
export const biasActivationSnippet = (hasBias, activation) => `
      ${hasBias ? 'value = value + getBiasByOutputCoords(coords);' : ''}
      ${activation ? 'value = activation(value, coords);' : ''}
      `;
