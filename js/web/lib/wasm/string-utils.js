// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
import {getInstance} from './wasm-factory.js';
/**
 * @param {string} data
 * @param {number[]} allocs
 * @returns {number}
 */
export const allocWasmString = (data, allocs) => {
  const wasm = getInstance();

  const dataLength = wasm.lengthBytesUTF8(data) + 1;
  const dataOffset = wasm._malloc(dataLength);
  wasm.stringToUTF8(data, dataOffset, dataLength);
  allocs.push(dataOffset);

  return dataOffset;
};
