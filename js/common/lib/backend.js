// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

//import {Backend} from './backend';

/**
 * @typedef {Object} BackendInfo
 * @property {Backend} backend
 * @property {number} priority
 * @property {Promise<void>} [initPromise]
 * @property {boolean} [initialized]
 * @property {boolean} [aborted]
 */

/** @type {{[name: string]: BackendInfo}} */
export const backends = {};
/** @type {string[]} */
export const backendsSortedByPriority = [];

/**
 * Register a backend.
 *
 * @param {string} name - the name as a key to lookup as an execution provider.
 * @param {Backend} backend - the backend object.
 * @param {number} priority - an integer indicating the priority of the backend. Higher number means higher priority. if priority
 * < 0, it will be considered as a 'beta' version and will not be used as a fallback backend by default.
 * @returns {void}
 * @internal
 */
export const registerBackend = (name, backend, priority) => {
  if (backend && typeof backend.init === 'function' && typeof backend.createSessionHandler === 'function') {
    const currentBackend = backends[name];
    if (currentBackend === undefined) {
      backends[name] = {backend, priority};
    } else if (currentBackend.priority > priority) {
      // same name is already registered with a higher priority. skip registeration.
      return;
    } else if (currentBackend.priority === priority) {
      if (currentBackend.backend !== backend) {
        throw new Error(`cannot register backend "${name}" using priority ${priority}`);
      }
    }

    if (priority >= 0) {
      const i = backendsSortedByPriority.indexOf(name);
      if (i !== -1) {
        backendsSortedByPriority.splice(i, 1);
      }

      for (let i = 0; i < backendsSortedByPriority.length; i++) {
        if (backends[backendsSortedByPriority[i]].priority <= priority) {
          backendsSortedByPriority.splice(i, 0, name);
          return;
        }
      }
      backendsSortedByPriority.push(name);
    }
    return;
  }

  throw new TypeError('not a valid backend');
};

/**
 * Resolve backend by specified hints.
 *
 * @param {readonly string[]} backendHints - a list of execution provider names to lookup. If omitted use registered backends as list.
 * @returns {Promise<Backend>} a promise that resolves to the backend.
 *
 * @internal
 */
export const resolveBackend = async(backendHints) => {
  const backendNames = backendHints.length === 0 ? backendsSortedByPriority : backendHints;
  const errors = [];
  for (const backendName of backendNames) {
    const backendInfo = backends[backendName];
    if (backendInfo) {
      if (backendInfo.initialized) {
        return backendInfo.backend;
      } else if (backendInfo.aborted) {
        continue;  // current backend is unavailable; try next
      }

      const isInitializing = !!backendInfo.initPromise;
      try {
        //debugger;
        if (!isInitializing) {
          backendInfo.initPromise = backendInfo.backend.init();
        }
        await backendInfo.initPromise;
        backendInfo.initialized = true;
        return backendInfo.backend;
      } catch (e) {
        if (!isInitializing) {
          errors.push({name: backendName, err: e});
        }
        backendInfo.aborted = true;
      } finally {
        delete backendInfo.initPromise;
      }
    }
  }

  throw new Error(`no available backend found. ERR: ${errors.map(e => `[${e.name}] ${e.err}`).join(', ')}`);
};
