import { type Page } from 'playwright'
import { LanguageModelV2 } from '@ai-sdk/provider'
import { z } from 'zod'
import { Schema as AiSchema } from 'ai'

import { preprocess, PreProcessOptions } from './preprocess.js'
import {
  generateAISDKCompletions,
  streamAISDKCompletions,
  generateAISDKCode,
} from './models.js'

// Options for high-level LLM calls
export type ScraperLLMOptions = {
  prompt?: string
  temperature?: number
  maxOutputTokens?: number
  topP?: number
  mode?: 'auto' | 'json' | 'tool'
}

// Options for code generation
export type ScraperGenerateOptions = Omit<
  ScraperLLMOptions,
  'mode'
> & {
  format?: 'html' | 'raw_html'
}

// Combined options for running scraper
export type ScraperRunOptions = ScraperLLMOptions & PreProcessOptions

// Union schema type that matches models.ts
export type ZodOrAiSchema<T> = z.ZodType<T> | AiSchema<T>

export default class LLMScraper {
  constructor(private client: LanguageModelV2) {
    this.client = client
  }

  // Run the scraper end-to-end
  async run<T>(
    page: Page,
    schema: ZodOrAiSchema<T>,
    options?: ScraperRunOptions
  ) {
    const preprocessed = await preprocess(page, options)
    return generateAISDKCompletions(
      this.client,
      preprocessed,
      schema,
      options
    )
  }

  // Stream partial results from the scraper
  async stream<T>(
    page: Page,
    schema: ZodOrAiSchema<T>,
    options?: ScraperRunOptions
  ) {
    const preprocessed = await preprocess(page, options)
    return streamAISDKCompletions(
      this.client,
      preprocessed,
      schema,
      options
    )
  }

  // Generate scraping code instead of data
  async generate<T>(
    page: Page,
    schema: ZodOrAiSchema<T>,
    options?: ScraperGenerateOptions
  ) {
    const preprocessed = await preprocess(page, options)
    return generateAISDKCode(
      this.client,
      preprocessed,
      schema,
      options
    )
  }
}
