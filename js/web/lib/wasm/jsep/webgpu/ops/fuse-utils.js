// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import {MAX_CLIP, MIN_CLIP} from '../../util.js';
/*
export interface InternalActivationAttributes {
  readonly activation: string;
  readonly clipMin?: number;
  readonly clipMax?: number;
  readonly activationCacheKey: string;
}
*/
/**
 *
 * @param {InternalActivationAttributes} attributes
 * @returns {{activationFunction: string; applyActivation: string}}
 */
export const getActicationSnippet = (attributes) => {
      switch (attributes.activation) {
        case 'Relu':
          return {activationFunction: '', applyActivation: 'value = max(value, 0.0);'};
        case 'Sigmoid':
          return {activationFunction: '', applyActivation: 'value = (1.0 / (1.0 + exp(-value)));'};
        case 'Clip':
          return {
            activationFunction:
                `const clip_min_=f32(${attributes.clipMin/*!*/});const clip_max_=f32(${attributes.clipMax/*!*/});`,
            applyActivation: 'value = clamp(value, clip_min_, clip_max_);'
          };
          // TODO: adding other activations that can be fused.
        default:
          return {activationFunction: '', applyActivation: ''};
      }
    };
/**
 *
 * @param {Record<string, unknown>|undefined} attributes
 * @returns {InternalActivationAttributes}
 */
export const parseInternalActivationAttributes = (attributes) => {
      const activation = attributes?.activation /*as string*/ || '';
      if (activation === 'Clip') {
        const [clipMin, clipMax] = attributes?.activation_params /*as [number, number]*/ || [MIN_CLIP, MAX_CLIP];
        return {activation, clipMax, clipMin, activationCacheKey: `${activation}:${clipMin},${clipMax}`};
      }
      return {activation, activationCacheKey: activation};
    };
