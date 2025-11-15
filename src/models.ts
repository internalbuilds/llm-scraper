import { LanguageModelV2 } from '@ai-sdk/provider'
import {
  generateObject,
  generateText,
  streamObject,
  UserContent,
  Schema as AiSchema,
} from 'ai'
import { z, ZodTypeAny } from 'zod'
import { ScraperLLMOptions, ScraperGenerateOptions } from './index.js'
import { PreProcessResult } from './preprocess.js'
import { zodToJsonSchema } from '@alcyone-labs/zod-to-json-schema'

// Unified schema type
type ZodOrAiSchema = ZodTypeAny | AiSchema<any>

const defaultPrompt =
  'You are a sophisticated web scraper. Extract the contents of the webpage'

const defaultCodePrompt =
  "Provide a scraping function in JavaScript that extracts and returns data according to a schema from the current page. The function must be IIFE. No comments or imports. No console.log. The code you generate will be executed straight away, you shouldn't output anything besides runnable code."

function stripMarkdownBackticks(text: string) {
  let trimmed = text.trim()
  trimmed = trimmed.replace(/^```(?:javascript)?\s*/i, '')
  trimmed = trimmed.replace(/\s*```$/i, '')
  return trimmed
}

function prepareAISDKPage(page: PreProcessResult): UserContent {
  if (page.format === 'image') {
    return [
      {
        type: 'image',
        image: page.content,
      },
    ]
  }
  return [{ type: 'text', text: page.content }]
}

export async function generateAISDKCompletions(
  model: LanguageModelV2,
  page: PreProcessResult,
  schema: ZodOrAiSchema,
  options?: ScraperLLMOptions
) {
  const content = prepareAISDKPage(page)

  const result = await generateObject<any>({
    model,
    messages: [
      { role: 'system', content: options?.prompt || defaultPrompt },
      { role: 'user', content },
    ],
    schema,
    temperature: options?.temperature,
    maxOutputTokens: options?.maxOutputTokens,
    topP: options?.topP,
    mode: options?.mode,
  })

  return {
    data: result.object,
    url: page.url,
  }
}

export function streamAISDKCompletions(
  model: LanguageModelV2,
  page: PreProcessResult,
  schema: ZodOrAiSchema,
  options?: ScraperLLMOptions
) {
  const content = prepareAISDKPage(page)

  const { partialObjectStream } = streamObject<any>({
    model,
    messages: [
      { role: 'system', content: options?.prompt || defaultPrompt },
      { role: 'user', content },
    ],
    schema,
    temperature: options?.temperature,
    maxOutputTokens: options?.maxOutputTokens,
    topP: options?.topP,
    mode: options?.mode,
  })

  return {
    stream: partialObjectStream,
    url: page.url,
  }
}

export async function generateAISDKCode(
  model: LanguageModelV2,
  page: PreProcessResult,
  schema: ZodOrAiSchema,
  options?: ScraperGenerateOptions
) {
  const parsedSchema =
    schema instanceof z.ZodType ? zodToJsonSchema(schema) : schema

  const result = await generateText({
    model,
    messages: [
      { role: 'system', content: options?.prompt || defaultCodePrompt },
      {
        role: 'user',
        content: `Website: ${page.url}
        Schema: ${JSON.stringify(parsedSchema)}
        Content: ${page.content}`,
      },
    ],
    temperature: options?.temperature,
    maxOutputTokens: options?.maxOutputTokens,
    topP: options?.topP,
  })

  return {
    code: stripMarkdownBackticks(result.text),
    url: page.url,
  }
}
