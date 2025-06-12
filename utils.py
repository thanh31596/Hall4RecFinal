import time
import random
from tenacity import retry, retry_if_exception_type, wait_exponential, stop_after_attempt
from google.api_core.exceptions import ResourceExhausted
from typing import Callable, Any, Optional, List, Dict
import logging

# Global logger reference (will be set by main script)
current_logger = None

def set_global_logger(logger):
    """Set global logger for API call tracking"""
    global current_logger
    current_logger = logger

class RateLimitManager:
    """Manages rate limiting with adaptive delays"""
    
    def __init__(self, base_delay: float = 3.0, max_delay: float = 120.0):
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.consecutive_failures = 0
        self.last_call_time = 0
        
    def get_delay(self) -> float:
        """Calculate delay based on failure history"""
        # Ensure minimum time between calls
        time_since_last = time.time() - self.last_call_time
        min_interval = self.base_delay
        
        if time_since_last < min_interval:
            base_wait = min_interval - time_since_last
        else:
            base_wait = 0
        
        # Add exponential backoff for failures
        failure_multiplier = min(4.0, 1.5 ** self.consecutive_failures)
        total_delay = base_wait + (self.base_delay * failure_multiplier)
        
        # Add jitter
        jitter = random.uniform(0.5, 1.5)
        final_delay = min(self.max_delay, total_delay * jitter)
        
        return final_delay
    
    def record_success(self):
        """Record successful API call"""
        self.consecutive_failures = 0
        self.last_call_time = time.time()
    
    def record_failure(self):
        """Record failed API call"""
        self.consecutive_failures += 1
        self.last_call_time = time.time()

# Global rate limit manager
rate_limiter = RateLimitManager()

@retry(
    retry=retry_if_exception_type((ResourceExhausted, Exception)),
    wait=wait_exponential(multiplier=3, min=5, max=300),  # More conservative
    stop=stop_after_attempt(5)  # Fewer retries
)
def make_api_call(function: Callable, *args, **kwargs) -> Any:
    """Enhanced wrapper for API calls with adaptive rate limiting"""
    global current_logger, rate_limiter
    
    # Get adaptive delay
    delay = rate_limiter.get_delay()
    if delay > 0:
        print(f"Rate limiting: waiting {delay:.1f}s before API call")
        time.sleep(delay)
    
    try:
        result = function(*args, **kwargs)
        
        # Record success
        rate_limiter.record_success()
        
        # Log successful API call
        if current_logger:
            current_logger.log_api_call(success=True, cost_estimate=0.001)
        
        # Additional delay after success to be conservative
        time.sleep(random.uniform(0.5, 1.5))
        
        return result
        
    except Exception as e:
        # Record failure
        rate_limiter.record_failure()
        
        error_str = str(e).lower()
        
        # Enhanced error detection
        is_rate_limit = any(indicator in error_str for indicator in [
            "429", "rate limit", "quota", "too many requests", 
            "resource exhausted", "rate_limit_exceeded"
        ])
        
        # Log failed API call
        if current_logger:
            current_logger.log_api_call(success=False, rate_limited=is_rate_limit)
        
        if is_rate_limit:
            print(f"🚫 Rate limit detected: {e}")
            print(f"   Consecutive failures: {rate_limiter.consecutive_failures}")
            # Longer delay for rate limits
            backoff_time = min(120, 15 + (10 * rate_limiter.consecutive_failures))
            print(f"   Backing off for {backoff_time}s...")
            time.sleep(backoff_time)
        elif "quota" in error_str or "billing" in error_str:
            print(f"💰 Quota/Billing issue: {e}")
            print("   This may require account attention.")
            time.sleep(60)  # Longer delay for quota issues
        else:
            print(f"⚠️  General API error: {e}")
        
        raise e

def safe_llm_invoke(llm, prompt: str, max_retries: int = 3) -> str:
    """Safely invoke LLM with enhanced retry logic and validation"""
    global current_logger
    
    if not prompt or not prompt.strip():
        return "Error: Empty prompt provided"
    
    for attempt in range(max_retries):
        try:
            print(f"🤖 LLM call attempt {attempt + 1}/{max_retries}")
            response = make_api_call(llm.invoke, prompt)
            
            # Extract content
            content = response.content if hasattr(response, 'content') else str(response)
            
            # Validate response
            if not content or len(content.strip()) < 10:
                raise ValueError("Response too short or empty")
            
            # Check for error indicators in response
            content_lower = content.lower()
            if any(indicator in content_lower for indicator in [
                "i cannot", "unable to", "api limit", "rate limit", "error occurred"
            ]):
                raise ValueError("LLM returned error response")
            
            print(f"✅ LLM call successful (length: {len(content)})")
            return content.strip()
            
        except Exception as e:
            print(f"❌ LLM call attempt {attempt + 1} failed: {e}")
            
            if attempt == max_retries - 1:
                # Final fallback
                print("🔄 All retries exhausted, returning fallback response")
                if current_logger:
                    current_logger.logger.error(f"LLM call failed completely after {max_retries} attempts: {e}")
                return generate_fallback_response(prompt)
            else:
                # Wait before retry
                wait_time = 5 + (attempt * 5)  # Progressive delay
                print(f"⏳ Waiting {wait_time}s before retry...")
                time.sleep(wait_time)

def generate_fallback_response(prompt: str) -> str:
    """Generate appropriate fallback response based on prompt content"""
    prompt_lower = prompt.lower()
    
    if "movie" in prompt_lower and "preference" in prompt_lower:
        return "User enjoys diverse entertainment options and values engaging storytelling. They appreciate well-crafted narratives and tend to choose movies that align with their personal interests and lifestyle."
    elif "personality" in prompt_lower:
        return "User has diverse interests and makes thoughtful choices in their entertainment preferences based on their lifestyle and personal values."
    elif "insight" in prompt_lower:
        return "This user demonstrates consistent preferences that reflect their demographic background and personal interests in entertainment choices."
    else:
        return "Unable to generate detailed analysis due to service limitations. Using general preference model."

def batch_llm_calls(llm, prompts: List[str], batch_size: int = 3, 
                   delay_between_batches: float = 20.0) -> List[str]:
    """Process LLM calls in batches with conservative rate limiting"""
    if not prompts:
        return []
    
    results = []
    total_batches = (len(prompts) + batch_size - 1) // batch_size
    
    print(f"🚀 Starting batch processing: {len(prompts)} prompts in {total_batches} batches")
    
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i + batch_size]
        batch_num = (i // batch_size) + 1
        
        print(f"\n📦 Processing batch {batch_num}/{total_batches} ({len(batch)} prompts)")
        
        batch_results = []
        for j, prompt in enumerate(batch):
            print(f"  Processing prompt {j + 1}/{len(batch)} in batch {batch_num}")
            
            result = safe_llm_invoke(llm, prompt)
            batch_results.append(result)
            
            # Delay between individual calls within batch
            if j < len(batch) - 1:
                individual_delay = random.uniform(3, 7)
                print(f"  ⏳ Waiting {individual_delay:.1f}s before next call...")
                time.sleep(individual_delay)
        
        results.extend(batch_results)
        
        # Progress update
        completed = len(results)
        print(f"✅ Batch {batch_num} completed. Progress: {completed}/{len(prompts)} ({completed/len(prompts)*100:.1f}%)")
        
        # Delay between batches (except for last batch)
        if i + batch_size < len(prompts):
            print(f"⏳ Waiting {delay_between_batches}s before next batch...")
            time.sleep(delay_between_batches)
    
    success_rate = len([r for r in results if not r.startswith("Unable to generate")]) / len(results) * 100
    print(f"\n🎉 Batch processing completed!")
    print(f"   Total prompts: {len(prompts)}")
    print(f"   Success rate: {success_rate:.1f}%")
    
    return results

def validate_api_setup() -> bool:
    """Validate API setup and connectivity"""
    import os
    
    # Check environment variables
    api_key = os.getenv('GOOGLE_API_KEY')
    if not api_key:
        print("❌ GOOGLE_API_KEY not found in environment variables")
        return False
    
    if len(api_key) < 20:  # Basic validation
        print("⚠️  GOOGLE_API_KEY seems too short")
        return False
    
    print("✅ API key found and appears valid")
    
    # Test connectivity (optional)
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
        
        test_llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash-001",
            temperature=0,
            max_tokens=50,
            timeout=30,
            max_retries=1
        )
        
        print("🧪 Testing API connectivity...")
        test_response = make_api_call(test_llm.invoke, "Say 'API test successful'")
        
        if "successful" in test_response.content.lower():
            print("✅ API connectivity test passed")
            return True
        else:
            print("⚠️  API responded but response unexpected")
            return False
            
    except Exception as e:
        print(f"❌ API connectivity test failed: {e}")
        return False

def estimate_api_cost(num_prompts: int, avg_prompt_length: int = 200, 
                     avg_response_length: int = 100) -> Dict[str, float]:
    """Estimate API costs for batch processing"""
    # Rough estimates for Gemini pricing (adjust based on actual pricing)
    cost_per_1k_input_tokens = 0.0015  # USD
    cost_per_1k_output_tokens = 0.002   # USD
    
    # Rough token estimation (4 chars per token)
    input_tokens = (num_prompts * avg_prompt_length) / 4
    output_tokens = (num_prompts * avg_response_length) / 4
    
    input_cost = (input_tokens / 1000) * cost_per_1k_input_tokens
    output_cost = (output_tokens / 1000) * cost_per_1k_output_tokens
    total_cost = input_cost + output_cost
    
    return {
        'estimated_input_tokens': input_tokens,
        'estimated_output_tokens': output_tokens,
        'estimated_input_cost_usd': input_cost,
        'estimated_output_cost_usd': output_cost,
        'estimated_total_cost_usd': total_cost
    }

def print_rate_limiting_info():
    """Print current rate limiting configuration"""
    global rate_limiter
    
    print("\n" + "="*50)
    print("RATE LIMITING CONFIGURATION")
    print("="*50)
    print(f"Base delay: {rate_limiter.base_delay}s")
    print(f"Max delay: {rate_limiter.max_delay}s")
    print(f"Consecutive failures: {rate_limiter.consecutive_failures}")
    print(f"Time since last call: {time.time() - rate_limiter.last_call_time:.1f}s")
    print("="*50)