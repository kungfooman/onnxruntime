// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

class AttributeWithCacheKeyImpl {
  /**
   * @param {Record<string, unknown>} attribute
   */
  constructor(attribute) {
    Object.assign(this, attribute);
  }

  /**
   * @type {string}
   * @private
   */
  _cacheKey;

  /**
   * @type {string}
   */
  get cacheKey() {
    if (!this._cacheKey) {
      this._cacheKey =
          Object.getOwnPropertyNames(this).sort().map(name => `${(this /*as Record<string, unknown>*/)[name]}`).join(';');
    }
    return this._cacheKey;
  }
}
/*
export interface AttributeWithCacheKey {
  readonly cacheKey: string;
}
*/
/**
 * create a new object from the given attribute, and add a cacheKey property to it
 *
 * @param {T} attribute
 * @template {Record<string, unknown>} T
 * @returns {T&AttributeWithCacheKey}
 */
export const createAttributeWithCacheKey = (attribute) =>
  new AttributeWithCacheKeyImpl(attribute) /*as unknown as T & AttributeWithCacheKey*/;
