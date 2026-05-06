# Full Conversational AI Setup - Gemini Agent

## Overview

This guide sets up the **complete conversational AI capability** using Google Gemini 2.0 Flash with:
- ✅ Natural conversation flow
- ✅ Conditional branching based on answers
- ✅ Early exclusion detection
- ✅ Iterative compliance checking
- ✅ Dynamic question prioritization
- ✅ 95% confidence threshold

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Consumer Input                           │
│              "I have some savings to invest"                 │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                  Gemini 2.0 Flash LLM                       │
│            (Natural Language Understanding)                  │
│  - Interprets consumer intent                               │
│  - Asks contextual follow-up questions                      │
│  - Handles conversational nuances                           │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│              ADK Tools (record_answer, etc.)                │
│  After EACH answer:                                         │
│  1. Update TraitGraph                                       │
│  2. Run ML prediction (automatic)                           │
│  3. Check excluding characteristics                         │
│  4. Check compliance rules                                  │
│  5. Decide next question OR stop                            │
└────────────────────────┬────────────────────────────────────┘
                         │
         ┌───────────────┴───────────────┐
         │                               │
         ▼                               ▼
    Continue?                        Stop?
 (confidence < 95%)              (confidence ≥ 95%
                                  OR excluded
                                  OR complete)
```

---

## Prerequisites

### 1. Google Cloud Project

```bash
# Install gcloud CLI
brew install google-cloud-sdk  # macOS
# or visit https://cloud.google.com/sdk/docs/install

# Login
gcloud auth login
gcloud auth application-default login

# Create project (or use existing)
export PROJECT_ID="ts-agent-demo"
gcloud projects create $PROJECT_ID
gcloud config set project $PROJECT_ID

# Enable required APIs
gcloud services enable aiplatform.googleapis.com
gcloud services enable vertexai.googleapis.com
```

### 2. Install Dependencies

```bash
cd /Users/iArk/Documents/ts-agent-e2e/ts_agent
source .venv/bin/activate

# Install ADK and dependencies
uv pip install google-adk==1.31.1
uv pip install google-cloud-aiplatform==1.70.0

# Verify
python -c "from google.adk.llms import VertexAILlm; print('ADK Ready!')"
```

### 3. Environment Configuration

```bash
# Set environment variables
export GOOGLE_CLOUD_PROJECT="your-project-id"
export VERTEX_LOCATION="europe-west2"  # Or your preferred region
export GEMINI_MODEL="gemini-2.0-flash"
export TS_ML_CONFIDENCE_THRESHOLD="0.95"

# Add to ~/.zshrc or ~/.bashrc for persistence
echo 'export GOOGLE_CLOUD_PROJECT="your-project-id"' >> ~/.zshrc
echo 'export VERTEX_LOCATION="europe-west2"' >> ~/.zshrc
echo 'export GEMINI_MODEL="gemini-2.0-flash"' >> ~/.zshrc
```

---

## Agent Capabilities

### 1. Natural Conversation

**Before (Robotic):**
```
Q: Do you currently hold any investment products (S&S ISA, funds, GIA)?
```

**After (Conversational AI):**
```
Agent: "Hello! I'm here to help you find financial support that's 
        right for you. To start, can I ask - do you currently have 
        any investment accounts, like an ISA or investment funds?"

Consumer: "Yes, I have a small ISA"

Agent: "That's great! Can you tell me roughly how much you have in 
        your ISA?"

Consumer: "About £5,000"

Agent: "Thanks! And when did you open it?"

Consumer: "About 2 years ago"

Agent: [Records: has_investments=true, investment_balance=5000, 
        investment_age_months=24]
       [Checks: SEG-INV-005 (Dormant) vs SEG-INV-001 (New investor)]
```

### 2. Conditional Branching

```python
# Gemini automatically branches based on context

# Scenario A: Has investments
Consumer: "Yes, I have investments"
Agent: "Can you tell me what type?"
       → Leads to dormant account check
       → May SUPPRESS if already adequately served

# Scenario B: No investments  
Consumer: "No, I don't"
Agent: "No problem! That means we can look at first-time 
        investor options. How much do you typically save?"
       → Leads to first-time investor path
       → Checks savings level
```

### 3. Early Exclusion Detection

```python
# Example: High-cost debt
Consumer: "I want to invest my £10k"
Agent: "Great! Before we continue, I need to check - do you 
        have any high-cost borrowing, like an overdraft or 
        payday loan?"

Consumer: "Yes, I have a £2k overdraft"

Agent: ❌ [EXCLUDE IMMEDIATELY]
       "Thank you for letting me know. Because you have 
        high-cost debt, we can't offer investment suggestions 
        right now. I'd recommend speaking with MoneyHelper at 
        moneyhelper.org.uk about managing your debt first."
       
       [SUPPRESS - No more questions asked]
```

### 4. Iterative Compliance

```python
# After EVERY answer:

turn_1:
    answer: "No investments"
    → predict: SEG-INV-002 (45% conf)
    → check_exclusions: None
    → continue: True
    
turn_2:
    answer: "Age 28"
    → predict: SEG-INV-002 (68% conf)
    → check_exclusions: None
    → check_age: ≥18 ✓
    → continue: True

turn_3:
    answer: "£400 monthly surplus"
    → predict: SEG-INV-002 (83% conf)
    → check_exclusions: None
    → check_surplus: positive ✓
    → continue: True

turn_4:
    answer: "Risk tolerance 3/5"
    → predict: SEG-INV-002 (96% conf) ✅
    → check_exclusions: None
    → confidence ≥ 95%: STOP
    → match_segment: SEG-INV-002
```

---

## System Prompts

### Zone 2 Agent System Prompt

```
You are the LBG Targeted Support conversational assistant.

Your mission is to have a natural, friendly conversation to understand 
a consumer's financial situation, while the system handles ML prediction 
and compliance checking automatically in the background.

=== CONVERSATION STYLE ===

1. **Warm & Professional**
   - "Hello! I'm here to help..."
   - "Thanks for sharing that..."
   - "That's really helpful to know..."

2. **One Question at a Time**
   - Never ask multiple things at once
   - Let them finish before moving on
   - Acknowledge their answer first

3. **Natural Flow**
   - Don't sound like a form
   - Use conversational language
   - Show empathy and understanding

4. **Handle Uncertainty**
   - If they say "I don't know" → record as "unknown"
   - If unclear → ask for clarification
   - Never pressure for answers

=== EXCLUSION AWARENESS ===

🚨 CRITICAL: Stop IMMEDIATELY if consumer mentions:
- High-cost debt (overdraft, payday loan, credit card arrears)
- Vulnerability (health, care needs, financial difficulty)
- Already holds suitable products
- Lump sum > £75k (exit Targeted Support)

When excluded, be empathetic:
"Thank you for sharing that with me. Based on what you've told me, 
I'd recommend speaking with a specialist advisor. You can reach them 
at [advisor_url]"

=== TOOLS YOU HAVE ===

1. `record_consumer_answer` - Save their response
   → Automatically runs ML prediction
   → Automatically checks exclusions
   → Returns next suggested question

2. `check_graph_completeness` - See if we have enough info
   → Returns confidence level
   → Returns if ready to match

3. `match_segment` - Finalize the segment match
   → Only call when confidence ≥ 95% or graph complete

=== DECISION FLOW ===

AFTER EACH ANSWER:
1. Call record_consumer_answer
2. Check the response:
   - `excluded=true` → STOP & explain
   - `ready_for_match=true` → Call match_segment
   - Otherwise → Ask next question

3. Use suggested_traits to prioritize questions:
   - Most important first (ML-driven)
   - Natural conversation order
   - Don't jump topics abruptly

=== EXAMPLES ===

Good Opening:
"Hello! I'm here to help you find financial support that might be 
suitable for your situation. To start, can I ask a few questions 
about your current finances?"

Good Follow-up:
"Thanks! That's helpful. Based on what you've told me, I'd like 
to understand [next topic] better..."

Good Closure:
"Thank you so much for answering my questions! I have enough 
information now to find the right support for you. Let me check 
what we can offer..."

Remember: You're a helpful financial guide, not a robot!
```

---

## Running the Gemini Agent

### Option 1: Interactive Script

```bash
python run_zone2_with_gemini.py
```

**Features:**
- Real Gemini LLM conversation
- Natural language Q&A
- Automatic prediction after each answer
- Early stopping at 95%
- Complete audit trail

### Option 2: Full Pipeline with Gemini

```bash
python run_gemini_full_pipeline.py
```

**What it does:**
1. Zone 0: DeBERTa intent classification (or manual)
2. Zone 1: Build TraitGraph from bank data
3. Zone 1.5: Initial ML prediction
4. **Zone 2: Gemini conversation** ← Natural AI
5. Zone 3: Compliance validation
6. Zone 4: Consumer message delivery

### Option 3: Python API

```python
from ts_agent.zones.zone2.agent_ml_auto import ML_AUTO_GAP_FILL_AGENT
from ts_agent.zones.zone1_graph_builder import TraitGraphBuilder
from google.adk.runner import create_agent_runner

# Build initial graph
builder = TraitGraphBuilder()
graph = builder.build_graph(
    party_ref="CUST-001",
    intent_id="INTENT-FIRST-TIME-INVESTOR",
    situation_id="SIT-INV-002",
    bank_data={
        "age": 28,
        "savings_balance": 8000,
        "monthly_surplus": 400,
    }
)

# Initialize Gemini agent
runner = create_agent_runner(ML_AUTO_GAP_FILL_AGENT)

# Start conversation
initial_state = {
    "ts_graph": graph.to_dict(),
    "ts_session_id": "session-001",
    "ts_turn": 0,
}

# Consumer message
user_message = "I'd like to start investing"

# Get AI response
response = await runner.run(
    user_message=user_message,
    state=initial_state,
)

print(f"Agent: {response.content}")
print(f"Next action: {response.tool_calls}")
```

---

## Enhanced Agent Features

### Intelligent Question Prioritization

The agent uses **4-tier prioritization**:

1. **SHAP Features** (ML-driven)
   - "Your age is important for risk assessment"
   - Based on current prediction

2. **Segment-Specific Config**
   - "For first-time investors, we need to know..."
   - From `segment_fill_priorities.py`

3. **Global Importance**
   - "Monthly surplus is always critical"
   - Universal priorities

4. **Standard Fill Order**
   - Default fallback
   - Node.fill_priority

### Conversation Memory

```python
# Agent remembers context
Turn 1: "Do you have investments?"
        → Answer: "Yes"

Turn 2: Agent REMEMBERS and asks:
        "What type of investments do you have?"
        (Not: "Do you have investments?" again!)

Turn 3: Agent CONTINUES thread:
        "And how much is invested?"
```

### Error Handling

```python
# Graceful handling of unclear inputs

Consumer: "Maybe? I think so?"
Agent: "No worries! Can you tell me a bit more about what 
        you mean? For example, do you have an ISA or any 
        investment funds?"

Consumer: "sdkfjhsdf"
Agent: "I didn't quite catch that. Could you rephrase?"

Consumer: [No response for 30s]
Agent: "Take your time! I'm here when you're ready."
```

---

## Testing the Conversational AI

### Test Scenario 1: Happy Path

```
Agent: "Hello! Can I ask about your investment experience?"
You: "I've never invested before"

Agent: "That's absolutely fine! Many people are in the same 
        situation. How much do you typically save each month?"
You: "About £400"

Agent: "Great! And roughly how much do you have saved right now?"
You: "Around £8,000"

Agent: "Perfect! One more question - what's your age?"
You: "28"

Agent: "Thank you! I have enough information. Based on what 
        you've shared, I can see you're interested in getting 
        started with investing. Let me find the right support..."

[96% confidence → Matched SEG-INV-002]
```

### Test Scenario 2: Early Exclusion

```
Agent: "Hello! To start, do you have any high-cost borrowing?"
You: "Yes, I have a payday loan"

Agent: "Thank you for being honest. Because you have high-cost 
        debt, I can't offer investment suggestions right now. 
        The best thing would be to speak with MoneyHelper about 
        managing this first. They're free and impartial: 
        moneyhelper.org.uk"

[SUPPRESS immediately - no more questions]
```

### Test Scenario 3: Conditional Branching

```
Agent: "Do you currently have any investments?"
You: "Yes, I have an ISA"

Agent: "That's good! Can you tell me when you last added money 
        or made changes to it?"
You: "About 2 years ago"

Agent: "I see. And roughly how much is in it?"
You: "£15,000"

Agent: "Thanks! It sounds like your ISA hasn't been actively 
        managed recently. Based on this, you might benefit from 
        reviewing your investment strategy..."

[Leads to dormant account re-engagement → SEG-INV-005]
```

---

## Monitoring & Debugging

### Enable Detailed Logging

```bash
export TS_LOG_LEVEL="DEBUG"
python run_zone2_with_gemini.py
```

### View EAMGP Signals

```python
# All signals logged:
- CONVERSATION_STARTED
- GAP_FILL_ANSWERED (after each answer)
- SEG_PREDICT_COMPLETE (after each prediction)
- EXCLUSION_DETECTED (if excluded)
- SEGMENT_MATCHED (when confident)
- CONVERSATION_COMPLETED
```

### Audit Trail

Every conversation creates:
- Full transcript
- Prediction history
- Confidence evolution
- Exclusion checks
- Compliance results
- Final decision rationale

---

## Cost Estimation

**Gemini 2.0 Flash Pricing** (as of 2026):
- Input: $0.075 per 1M tokens
- Output: $0.30 per 1M tokens

**Typical Conversation:**
- 5-8 questions
- ~2,000 input tokens
- ~500 output tokens
- **Cost: ~$0.00030 per conversation**

Very affordable for production!

---

## Production Checklist

Before going live:

- [ ] GCP project created & configured
- [ ] Gemini API enabled
- [ ] Environment variables set
- [ ] ADK dependencies installed
- [ ] System prompts reviewed & approved
- [ ] Exclusion logic tested
- [ ] Compliance checks validated
- [ ] Audit logging configured
- [ ] Cost monitoring set up
- [ ] Fallback to human advisor configured

---

## Next Steps

1. **Setup GCP** (5 minutes)
   ```bash
   gcloud auth login
   export GOOGLE_CLOUD_PROJECT="your-project"
   ```

2. **Test Basic Agent** (2 minutes)
   ```bash
   python -c "from ts_agent.zones.zone2.agent_ml_auto import ML_AUTO_GAP_FILL_AGENT; print('Agent loaded!')"
   ```

3. **Run Interactive Demo** (Try it!)
   ```bash
   python run_zone2_with_gemini.py
   ```

4. **Review Conversation Logs**
   - Check EAMGP signals
   - Verify predictions
   - Confirm exclusions work

5. **Iterate & Improve**
   - Refine system prompts
   - Adjust confidence threshold
   - Add domain-specific nuances

---

## Support

**Issues?**
- Check `TROUBLESHOOTING.md`
- Review logs in `logs/gemini_conversations/`
- Contact: devops@lloydsbanking.com

**Ready to start?**
```bash
python run_zone2_with_gemini.py
```

🚀 **Your conversational AI is ready!**
