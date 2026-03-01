# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Utility functions for interacting with Gemini and Claude APIs, image processing, and PDF handling.
"""

import json
import asyncio
import base64
from io import BytesIO
from functools import partial
from ast import literal_eval
from typing import List, Dict, Any

import httpx

import aiofiles
from PIL import Image
from google import genai
from google.genai import types
from anthropic import AsyncAnthropic
from openai import AsyncOpenAI

import os

import yaml
from pathlib import Path

# Load config
config_path = Path(__file__).parent.parent / "configs" / "model_config.yaml"
model_config = {}
if config_path.exists():
    with open(config_path, "r") as f:
        model_config = yaml.safe_load(f) or {}

def get_config_val(section, key, env_var, default=""):
    val = os.getenv(env_var)
    if not val and section in model_config:
        val = model_config[section].get(key)
    return val or default

# Initialize clients lazily or with robust defaults
api_key = get_config_val("api_keys", "google_api_key", "GOOGLE_API_KEY", "")
if api_key:
    gemini_client = genai.Client(api_key=api_key)
    print("Initialized Gemini Client with API Key")
else:
    print("Warning: Could not initialize Gemini Client. Missing credentials.")
    gemini_client = None


anthropic_api_key = get_config_val("api_keys", "anthropic_api_key", "ANTHROPIC_API_KEY", "")
if anthropic_api_key:
    anthropic_client = AsyncAnthropic(api_key=anthropic_api_key)
    print("Initialized Anthropic Client with API Key")
else:
    print("Warning: Could not initialize Anthropic Client. Missing credentials.")
    anthropic_client = None

openai_api_key = get_config_val("api_keys", "openai_api_key", "OPENAI_API_KEY", "")
if openai_api_key:
    openai_client = AsyncOpenAI(api_key=openai_api_key)
    print("Initialized OpenAI Client with API Key")
else:
    print("Warning: Could not initialize OpenAI Client. Missing credentials.")
    openai_client = None

doubao_api_key = get_config_val("api_keys", "doubao_api_key", "DOUBAO_API_KEY", "")
doubao_base_url = get_config_val("doubao", "base_url", "DOUBAO_BASE_URL", "https://ark.cn-beijing.volces.com/api/v3")
if doubao_api_key:
    doubao_client = AsyncOpenAI(api_key=doubao_api_key, base_url=doubao_base_url)
    print("Initialized Doubao Client with API Key")
else:
    print("Warning: Could not initialize Doubao Client. Missing credentials.")
    doubao_client = None



def _convert_to_gemini_parts(contents: List[Dict[str, Any]]) -> List[types.Part]:
    """
    Convert a generic content list to a list of Gemini's genai.types.Part objects.
    """
    gemini_parts = []
    for item in contents:
        if item.get("type") == "text":
            gemini_parts.append(types.Part.from_text(text=item["text"]))
        elif item.get("type") == "image":
            source = item.get("source", {})
            if source.get("type") == "base64":
                gemini_parts.append(
                    types.Part.from_bytes(
                        data=base64.b64decode(source["data"]),
                        mime_type=source["media_type"],
                    )
                )
    return gemini_parts


async def call_gemini_with_retry_async(
    model_name, contents, config, max_attempts=5, retry_delay=5, error_context=""
):
    """
    ASYNC: Call Gemini API with asynchronous retry logic.
    """
    if gemini_client is None:
        raise RuntimeError(
            "Gemini client was not initialized: missing Google API key. "
            "Please set GOOGLE_API_KEY in environment, or configure api_keys.google_api_key in configs/model_config.yaml."
        )

    result_list = []
    target_candidate_count = config.candidate_count
    # Gemini API max candidate count is 8. We will call multiple times if needed.
    if config.candidate_count > 8:
        config.candidate_count = 8

    current_contents = contents
    for attempt in range(max_attempts):
        try:
            # Use global client
            client = gemini_client

            # Convert generic content list to Gemini's format right before the API call
            gemini_contents = _convert_to_gemini_parts(current_contents)
            response = await client.aio.models.generate_content(
                model=model_name, contents=gemini_contents, config=config
            )

            # If we are using Image Generation models to generate images
            if (
                "nanoviz" in model_name
                or "image" in model_name
            ):
                raw_response_list = []
                if not response.candidates or not response.candidates[0].content.parts:
                    print(
                        f"[Warning]: Failed to generate image, retrying in {retry_delay} seconds..."
                    )
                    await asyncio.sleep(retry_delay)
                    continue

                # In this mode, we can only have one candidate
                for part in response.candidates[0].content.parts:
                    if part.inline_data:
                        # Append base64 encoded image data to raw_response_list
                        raw_response_list.append(
                            base64.b64encode(part.inline_data.data).decode("utf-8")
                        )
                        break

            # Otherwise, for text generation models
            else:
                raw_response_list = [
                    part.text
                    for candidate in response.candidates
                    for part in candidate.content.parts
                ]
            result_list.extend([r for r in raw_response_list if r.strip() != ""])
            if len(result_list) >= target_candidate_count:
                result_list = result_list[:target_candidate_count]
                break

        except Exception as e:
            context_msg = f" for {error_context}" if error_context else ""
            
            # Exponential backoff (capped at 30s)
            current_delay = min(retry_delay * (2 ** attempt), 30)
            
            print(
                f"Attempt {attempt + 1} for model {model_name} failed{context_msg}: {e}. Retrying in {current_delay} seconds..."
            )

            if attempt < max_attempts - 1:
                await asyncio.sleep(current_delay)
            else:
                print(f"Error: All {max_attempts} attempts failed{context_msg}")
                result_list = ["Error"] * target_candidate_count

    if len(result_list) < target_candidate_count:
        result_list.extend(["Error"] * (target_candidate_count - len(result_list)))
    return result_list

def _convert_to_claude_format(contents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Converts the generic content list to Claude's API format.
    Currently, the formats are identical, so this acts as a pass-through
    for architectural consistency and future-proofing.

    Claude API's format:
    [
        {"type": "text", "text": "some text"},
        {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": "..."}},
        ...
    ]
    """
    return contents


def _convert_to_openai_format(contents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Converts the generic content list (Claude format) to OpenAI's API format.
    
    Claude format:
    [
        {"type": "text", "text": "some text"},
        {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": "..."}},
        ...
    ]
    
    OpenAI format:
    [
        {"type": "text", "text": "some text"},
        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}},
        ...
    ]
    """
    openai_contents = []
    for item in contents:
        if item.get("type") == "text":
            openai_contents.append({"type": "text", "text": item["text"]})
        elif item.get("type") == "image":
            source = item.get("source", {})
            if source.get("type") == "base64":
                media_type = source.get("media_type", "image/jpeg")
                data = source.get("data", "")
                # OpenAI expects data URL format
                data_url = f"data:{media_type};base64,{data}"
                openai_contents.append({
                    "type": "image_url",
                    "image_url": {"url": data_url}
                })
    return openai_contents


async def call_claude_with_retry_async(
    model_name, contents, config, max_attempts=5, retry_delay=30, error_context=""
):
    """
    ASYNC: Call Claude API with asynchronous retry logic.
    This version efficiently handles input size errors by validating and modifying
    the content list once before generating all candidates.
    """
    system_prompt = config["system_prompt"]
    temperature = config["temperature"]
    candidate_num = config["candidate_num"]
    max_output_tokens = config["max_output_tokens"]
    response_text_list = []

    # --- Preparation Phase ---
    # Convert to the Claude-specific format and perform an initial optimistic resize.
    current_contents = contents

    # --- Validation and Remediation Phase ---
    # We loop until we get a single successful response, proving the input is valid.
    # Note that this check is required because Claude only has 128k / 256k context windows.
    # For Gemini series that support 1M, we do not need this step.
    is_input_valid = False
    for attempt in range(max_attempts):
        try:
            claude_contents = _convert_to_claude_format(current_contents)
            # Attempt to generate the very first candidate.
            first_response = await anthropic_client.messages.create(
                model=model_name,
                max_tokens=max_output_tokens,
                temperature=temperature,
                messages=[{"role": "user", "content": claude_contents}],
                system=system_prompt,
            )
            response_text_list.append(first_response.content[0].text)
            is_input_valid = True
            break

        except Exception as e:
            error_str = str(e).lower()
            context_msg = f" for {error_context}" if error_context else ""
            print(
                f"Validation attempt {attempt + 1} failed{context_msg}: {error_str}. Retrying in {retry_delay} seconds..."
            )
            if attempt < max_attempts - 1:
                await asyncio.sleep(retry_delay)

    # --- Sampling Phase ---
    if not is_input_valid:
        print(
            f"Error: All {max_attempts} attempts failed to validate the input{context_msg}. Returning errors."
        )
        return ["Error"] * candidate_num

    # We already have 1 successful candidate, now generate the rest.
    remaining_candidates = candidate_num - 1
    if remaining_candidates > 0:
        print(
            f"Input validated. Now generating remaining {remaining_candidates} candidates..."
        )
        valid_claude_contents = _convert_to_claude_format(current_contents)
        tasks = [
            anthropic_client.messages.create(
                model=model_name,
                max_tokens=max_output_tokens,
                temperature=temperature,
                messages=[
                    {"role": "user", "content": valid_claude_contents}
                ],
                system=system_prompt,
            )
            for _ in range(remaining_candidates)
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)
        for res in results:
            if isinstance(res, Exception):
                print(f"Error generating a subsequent candidate: {res}")
                response_text_list.append("Error")
            else:
                response_text_list.append(res.content[0].text)

    return response_text_list

async def call_openai_with_retry_async(
    model_name, contents, config, max_attempts=5, retry_delay=30, error_context=""
):
    """
    ASYNC: Call OpenAI API with asynchronous retry logic.
    This follows the same pattern as Claude's implementation.
    """
    system_prompt = config["system_prompt"]
    temperature = config["temperature"]
    candidate_num = config["candidate_num"]
    max_completion_tokens = config["max_completion_tokens"]
    response_text_list = []

    # --- Preparation Phase ---
    # Convert to the OpenAI-specific format
    current_contents = contents

    # --- Validation and Remediation Phase ---
    # We loop until we get a single successful response, proving the input is valid.
    is_input_valid = False
    for attempt in range(max_attempts):
        try:
            openai_contents = _convert_to_openai_format(current_contents)
            # Attempt to generate the very first candidate.
            first_response = await openai_client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": openai_contents}
                ],
                temperature=temperature,
                max_completion_tokens=max_completion_tokens,
            )
            # If we reach here, the input is valid.
            response_text_list.append(first_response.choices[0].message.content)
            is_input_valid = True
            break  # Exit the validation loop

        except Exception as e:
            error_str = str(e).lower()
            context_msg = f" for {error_context}" if error_context else ""
            print(
                f"Validation attempt {attempt + 1} failed{context_msg}: {error_str}. Retrying in {retry_delay} seconds..."
            )
            if attempt < max_attempts - 1:
                await asyncio.sleep(retry_delay)

    # --- Sampling Phase ---
    if not is_input_valid:
        print(
            f"Error: All {max_attempts} attempts failed to validate the input{context_msg}. Returning errors."
        )
        return ["Error"] * candidate_num

    # We already have 1 successful candidate, now generate the rest.
    remaining_candidates = candidate_num - 1
    if remaining_candidates > 0:
        print(
            f"Input validated. Now generating remaining {remaining_candidates} candidates..."
        )
        valid_openai_contents = _convert_to_openai_format(current_contents)
        tasks = [
            openai_client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": valid_openai_contents}
                ],
                temperature=temperature,
                max_completion_tokens=max_completion_tokens,
            )
            for _ in range(remaining_candidates)
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)
        for res in results:
            if isinstance(res, Exception):
                print(f"Error generating a subsequent candidate: {res}")
                response_text_list.append("Error")
            else:
                response_text_list.append(res.choices[0].message.content)

    return response_text_list


async def call_doubao_with_retry_async(
    model_name, contents, config, max_attempts=5, retry_delay=30, error_context=""
):
    """
    ASYNC: Call Doubao (豆包) API with asynchronous retry logic.
    Doubao uses an OpenAI-compatible API via the Volcengine Ark platform.
    """
    if doubao_client is None:
        raise RuntimeError(
            "Doubao client was not initialized: missing Doubao API key. "
            "Please set DOUBAO_API_KEY in environment, or configure api_keys.doubao_api_key in configs/model_config.yaml."
        )

    system_prompt = config["system_prompt"]
    temperature = config["temperature"]
    candidate_num = config["candidate_num"]
    max_output_tokens = config.get("max_output_tokens", config.get("max_completion_tokens", 4096))
    response_text_list = []

    # --- Preparation Phase ---
    current_contents = contents

    # --- Validation and Remediation Phase ---
    is_input_valid = False
    for attempt in range(max_attempts):
        try:
            doubao_contents = _convert_to_openai_format(current_contents)
            first_response = await doubao_client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": doubao_contents}
                ],
                temperature=temperature,
                max_tokens=max_output_tokens,
            )
            response_text_list.append(first_response.choices[0].message.content)
            is_input_valid = True
            break

        except Exception as e:
            error_str = str(e).lower()
            context_msg = f" for {error_context}" if error_context else ""
            current_delay = min(retry_delay * (2 ** attempt), 30)
            print(
                f"Validation attempt {attempt + 1} failed{context_msg}: {error_str}. Retrying in {current_delay} seconds..."
            )
            if attempt < max_attempts - 1:
                await asyncio.sleep(current_delay)

    # --- Sampling Phase ---
    if not is_input_valid:
        context_msg = f" for {error_context}" if error_context else ""
        print(
            f"Error: All {max_attempts} attempts failed to validate the input{context_msg}. Returning errors."
        )
        return ["Error"] * candidate_num

    remaining_candidates = candidate_num - 1
    if remaining_candidates > 0:
        print(
            f"Input validated. Now generating remaining {remaining_candidates} candidates..."
        )
        valid_doubao_contents = _convert_to_openai_format(current_contents)
        tasks = [
            doubao_client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": valid_doubao_contents}
                ],
                temperature=temperature,
                max_tokens=max_output_tokens,
            )
            for _ in range(remaining_candidates)
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)
        for res in results:
            if isinstance(res, Exception):
                print(f"Error generating a subsequent candidate: {res}")
                response_text_list.append("Error")
            else:
                response_text_list.append(res.choices[0].message.content)

    return response_text_list


async def call_openai_image_generation_with_retry_async(
    model_name, prompt, config, max_attempts=5, retry_delay=30, error_context=""
):
    """
    ASYNC: Call OpenAI Image Generation API (GPT-Image) with asynchronous retry logic.
    """
    size = config.get("size", "1536x1024")
    quality = config.get("quality", "high")
    background = config.get("background", "opaque")
    output_format = config.get("output_format", "png")
    
    # Base parameters for all models
    gen_params = {
        "model": model_name,
        "prompt": prompt,
        "n": 1,
        "size": size,
    }
    
    # Add GPT-Image specific parameters
    gen_params.update({
        "quality": quality,
        "background": background,
        "output_format": output_format,
    })

    for attempt in range(max_attempts):
        try:
            response = await openai_client.images.generate(**gen_params)
            
            # OpenAI images.generate returns a list of images in response.data
            if response.data and response.data[0].b64_json:
                return [response.data[0].b64_json]
            else:
                print(f"[Warning]: Failed to generate image via OpenAI, no data returned.")
                if attempt < max_attempts - 1:
                    await asyncio.sleep(retry_delay)
                continue

        except Exception as e:
            context_msg = f" for {error_context}" if error_context else ""
            print(
                f"Attempt {attempt + 1} for OpenAI image generation model {model_name} failed{context_msg}: {e}. Retrying in {retry_delay} seconds..."
            )

            if attempt < max_attempts - 1:
                await asyncio.sleep(retry_delay)
            else:
                print(f"Error: All {max_attempts} attempts failed{context_msg}")
                return ["Error"]

    return ["Error"]


async def call_doubao_image_generation_with_retry_async(
    model_name, prompt, config, max_attempts=5, retry_delay=30, error_context=""
):
    """
    ASYNC: Call Doubao (豆包) Image Generation API with asynchronous retry logic.
    Doubao uses an OpenAI-compatible API via the Volcengine Ark platform.
    The endpoint is at {base_url}/images/generations.
    """
    if doubao_client is None:
        raise RuntimeError(
            "Doubao client was not initialized: missing Doubao API key. "
            "Please set DOUBAO_API_KEY in environment, or configure api_keys.doubao_api_key in configs/model_config.yaml."
        )

    size = config.get("size", "1024x1024")
    guidance_scale = config.get("guidance_scale", 2.5)
    watermark = config.get("watermark", False)
    response_format = config.get("response_format", "b64_json")

    gen_params = {
        "model": model_name,
        "prompt": prompt,
        "n": 1,
        "size": size,
        "response_format": response_format,
    }
    # Add Doubao-specific parameters if supported
    if guidance_scale is not None:
        gen_params["extra_body"] = {
            "guidance_scale": guidance_scale,
            "watermark": watermark,
        }

    for attempt in range(max_attempts):
        try:
            response = await doubao_client.images.generate(**gen_params)

            if response.data and response.data[0].b64_json:
                return [response.data[0].b64_json]
            else:
                print(f"[Warning]: Failed to generate image via Doubao, no data returned.")
                if attempt < max_attempts - 1:
                    await asyncio.sleep(retry_delay)
                continue

        except Exception as e:
            context_msg = f" for {error_context}" if error_context else ""
            current_delay = min(retry_delay * (2 ** attempt), 60)
            print(
                f"Attempt {attempt + 1} for Doubao image generation model {model_name} failed{context_msg}: {e}. Retrying in {current_delay} seconds..."
            )

            if attempt < max_attempts - 1:
                await asyncio.sleep(current_delay)
            else:
                print(f"Error: All {max_attempts} attempts failed{context_msg}")
                return ["Error"]

    return ["Error"]


async def call_doubao_video_generation_with_retry_async(
    model_name, prompt, config, max_attempts=5, retry_delay=30, error_context=""
):
    """
    ASYNC: Call Doubao (豆包) Video Generation API with asynchronous retry logic.
    Uses the Volcengine Ark content generation API at {base_url}/contents/generations/tasks.
    This is an async task-based API: create a task, then poll for results.
    
    Returns a list containing the video URL on success, or ["Error"] on failure.
    """
    if not doubao_api_key:
        raise RuntimeError(
            "Doubao client was not initialized: missing Doubao API key. "
            "Please set DOUBAO_API_KEY in environment, or configure api_keys.doubao_api_key in configs/model_config.yaml."
        )

    base_url = doubao_base_url.rstrip("/")
    create_url = f"{base_url}/contents/generations/tasks"
    poll_interval = config.get("poll_interval", 5)
    max_poll_time = config.get("max_poll_time", 300)  # 5 minutes max polling

    content_items = [{"type": "text", "text": prompt}]

    payload = {
        "model": model_name,
        "content": content_items,
    }

    headers = {
        "Authorization": f"Bearer {doubao_api_key}",
        "Content-Type": "application/json",
    }

    for attempt in range(max_attempts):
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                # Step 1: Create the video generation task
                create_response = await client.post(
                    create_url, json=payload, headers=headers
                )
                create_response.raise_for_status()
                task_data = create_response.json()
                task_id = task_data.get("id")

                if not task_id:
                    print(f"[Warning]: Failed to create video generation task, no task ID returned.")
                    if attempt < max_attempts - 1:
                        await asyncio.sleep(retry_delay)
                    continue

                print(f"Video generation task created: {task_id}")

                # Step 2: Poll for task completion
                poll_url = f"{create_url}/{task_id}"
                elapsed = 0
                while elapsed < max_poll_time:
                    await asyncio.sleep(poll_interval)
                    elapsed += poll_interval

                    poll_response = await client.get(poll_url, headers=headers)
                    poll_response.raise_for_status()
                    result_data = poll_response.json()
                    status = result_data.get("status", "")

                    if status == "succeeded":
                        # Extract video URL from response content
                        content_list = result_data.get("content", [])
                        for item in content_list:
                            if item.get("type") == "video_url":
                                video_url = item.get("video_url", "")
                                if isinstance(video_url, dict):
                                    video_url = video_url.get("url", "")
                                if video_url:
                                    print(f"Video generation succeeded: {video_url}")
                                    return [video_url]
                        print(f"[Warning]: Video task succeeded but no video URL found in response.")
                        return ["Error"]

                    elif status == "failed":
                        error_info = result_data.get("error", "Unknown error")
                        print(f"Video generation task failed: {error_info}")
                        break

                    else:
                        print(f"Video generation status: {status} (elapsed: {elapsed}s)")

                if elapsed >= max_poll_time:
                    print(f"Video generation task timed out after {max_poll_time}s")

        except Exception as e:
            context_msg = f" for {error_context}" if error_context else ""
            current_delay = min(retry_delay * (2 ** attempt), 60)
            print(
                f"Attempt {attempt + 1} for Doubao video generation model {model_name} failed{context_msg}: {e}. Retrying in {current_delay} seconds..."
            )

            if attempt < max_attempts - 1:
                await asyncio.sleep(current_delay)
            else:
                print(f"Error: All {max_attempts} attempts failed{context_msg}")
                return ["Error"]

    return ["Error"]


async def call_text_model_with_retry_async(
    model_name, contents, system_prompt, temperature, candidate_num=1,
    max_output_tokens=50000, max_attempts=5, retry_delay=5, error_context=""
):
    """
    Unified text generation dispatcher that routes to the correct provider
    based on model_name. Supports Gemini, Claude, OpenAI, and Doubao.
    """
    if "gemini" in model_name:
        return await call_gemini_with_retry_async(
            model_name=model_name,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=temperature,
                candidate_count=candidate_num,
                max_output_tokens=max_output_tokens,
            ),
            max_attempts=max_attempts,
            retry_delay=retry_delay,
            error_context=error_context,
        )
    elif "doubao" in model_name:
        return await call_doubao_with_retry_async(
            model_name=model_name,
            contents=contents,
            config={
                "system_prompt": system_prompt,
                "temperature": temperature,
                "candidate_num": candidate_num,
                "max_output_tokens": max_output_tokens,
            },
            max_attempts=max_attempts,
            retry_delay=retry_delay,
            error_context=error_context,
        )
    elif "claude" in model_name:
        return await call_claude_with_retry_async(
            model_name=model_name,
            contents=contents,
            config={
                "system_prompt": system_prompt,
                "temperature": temperature,
                "candidate_num": candidate_num,
                "max_output_tokens": max_output_tokens,
            },
            max_attempts=max_attempts,
            retry_delay=retry_delay,
            error_context=error_context,
        )
    elif "gpt" in model_name or "o1" in model_name or "o3" in model_name or "o4" in model_name:
        return await call_openai_with_retry_async(
            model_name=model_name,
            contents=contents,
            config={
                "system_prompt": system_prompt,
                "temperature": temperature,
                "candidate_num": candidate_num,
                "max_completion_tokens": max_output_tokens,
            },
            max_attempts=max_attempts,
            retry_delay=retry_delay,
            error_context=error_context,
        )
    else:
        raise ValueError(f"Unsupported text model: {model_name}. Model name must contain 'gemini', 'doubao', 'claude', 'gpt', 'o1', 'o3', or 'o4'.")
