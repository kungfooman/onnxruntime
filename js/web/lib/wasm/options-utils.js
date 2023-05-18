// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

//interface ExtraOptionsHandler {
//  (name: string, value: string): void;
//}
/**
 *
 * @param {Record<string, unknown>} options
 * @param {string} prefix
 * @param {WeakSet<Record<string, unknown>>} seen
 * @param {ExtraOptionsHandler} handler
 * @returns {void}
 */
export const iterateExtraOptions = (options, prefix, seen, handler) => {
      if (typeof options == 'object' && options !== null) {
        if (seen.has(options)) {
          throw new Error('Circular reference in options');
        } else {
          seen.add(options);
        }
      }

      Object.entries(options).forEach(([key, value]) => {
        const name = (prefix) ? prefix + key : key;
        if (typeof value === 'object') {
          iterateExtraOptions(value /*as Record<string, unknown>*/, name + '.', seen, handler);
        } else if (typeof value === 'string' || typeof value === 'number') {
          handler(name, value.toString());
        } else if (typeof value === 'boolean') {
          handler(name, (value) ? '1' : '0');
        } else {
          throw new Error(`Can't handle extra config type: ${typeof value}`);
        }
      });
    };
