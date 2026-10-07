import random
import numpy as np
import pandas as pd

from src.scenarios.base import BaseScenario

PUBLIC_FIELDS = [
    "loan_amount",
    "loan_term",
    "loan_type",
    "loan_purpose",
    "preapproval"
]

_LOAN_TYPE_BULLET = "- **Loan Type**: The type of covered loan or application (e.g., conventional, FHA, VA)"
_LOAN_PURPOSE_BULLET = "- **Loan Purpose**: The purpose of covered loan or application (e.g., home purchase, refinance)"
_LOAN_AMOUNT_BULLET = "- **Loan Amount**: The requested loan amount in dollars"
_LOAN_TERM_BULLET = "- **Loan Term**: Duration of the loan in months"
_PREAPPROVAL_BULLET = (
    "- **Preapproval**: Whether the covered loan or application involved a request for a preapproval"
    " of a home purchase loan under a preapproval program"
)

PUBLIC_FIELD_DESCRIPTIONS = "\n".join(
    [_LOAN_TYPE_BULLET, _LOAN_PURPOSE_BULLET, _LOAN_AMOUNT_BULLET, _LOAN_TERM_BULLET, _PREAPPROVAL_BULLET]
)

AGENT_FIELD_DESCRIPTIONS = (
    "- **Action Taken**: The action taken on the covered loan or application\n"
    "- **Denial Reasons**: The principal reason, or reasons, for denial if applicable."
    " There can be up to four reasons the loan was denied\n"
    f"{_LOAN_TYPE_BULLET}\n"
    f"{_LOAN_PURPOSE_BULLET}\n"
    f"{_LOAN_AMOUNT_BULLET}\n"
    f"{_LOAN_TERM_BULLET}\n"
    "- **Interest Rate**: The offered interest rate as a percentage\n"
    "- **Debt-to-Income (DTI) Ratio**: The ratio, as a percentage, of the applicant's or borrower's"
    " total monthly debt to the total monthly income relied on in making the credit decision\n"
    "- **Total Loan Costs**: Total costs associated with the loan in dollars\n"
    "- **Property Value**: Estimated value of the property proposed to secure the covered loan in dollars\n"
    "- **HOEPA Status**: Whether the loan is classified as a high-cost mortgage under HOEPA\n"
    "- **Negative Amortization**: Whether the loan has negative amortization features\n"
    "- **Purchaser Type**: Whether and to whom the loan was sold after origination\n"
    f"{_PREAPPROVAL_BULLET}"
)

class LoanQAScenario(BaseScenario):
    """
    This provides the scenario structure for the loan qa, where the agent LLM must answer Q&A questions from "user" LLMs about the status of their home loans.
    "user" LLM.
    """

    def __init__(
        self,
        loan_file: str,
        goal: str,
        max_turns: int,
        seed: int | None = None,
    ):
        self.agent_goal = goal
        self.max_turns = max_turns
        self.seed = seed
        self._toolkit = None

        self.loan_data = pd.read_csv(loan_file)
        self.public_applicant_data = None
        self.full_applicant_data = None

    def setup(self, seed):
        """
        Allows for overridding the seed without re-initializing
        """
        self.seed = seed
        if self.seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        self._initialize_user()

    def stopping_conditions(self):
        """
        Uses a 5-point Likert scale for user satisfaction to end the interaction.
        """
        return ["<Very Satisfied>", "<Satisfied>", "<Neutral>", "<Dissatisfied>", "<Very Dissatisfied>"]

    def opening_messages(self) -> dict[str, str]:
        return {"agent": "Hello, how may I help you today?"}

    def system_prompts(self):
        scenario = self._initialize_scenario_text()

        agent_messages = [{
            "role": "system",
            "content": self._initialize_agent_sys_prompt(scenario)
        }]

        user_messages = [{
            "role": "system",
            "content": self._initialize_user_sys_prompt(scenario)
        }]

        return {"agent": agent_messages, "user": user_messages}

    def scenario_parameters(self):
        return {"applicant_profile": self.full_applicant_data, "agent_goal": self.agent_goal}

    def _initialize_scenario_text(self):
        """
        Initializes the scenario/context explanation of the system prompts.
        """
        stopping_conditions = self.stopping_conditions()
        scenario = f"""
        A customer service agent is responding to a consumer inquiring about the status
        and details of their housing loan application.

        The consumer may have questions about:
        - The outcome of their loan application
        - Their loan terms and costs
        - The reason for any denial or unfavorable terms
        - Whether their loan has been sold to another institution

        The consumer only knows they submitted an application and are awaiting or
        have received a response. They do not have access to the full loan record.

        Only the agent has access to the complete loan record, including action taken,
        denial reasons, rate details, high-cost flags, and purchaser information.

        The consumer will rate the interaction based on the following service quality
        dimensions:
        - Competence: did the agent demonstrate accurate knowledge of the loan details
        - Courtesy: did the agent engage with politeness, respect, and professionalism
        - Credibility: did the agent appear trustworthy and act in the consumer's
        best interest
        - Security: did the agent make the consumer feel safe from doubt, risk,
        or financial confusion
        - Communication: did the agent explain information clearly in plain language
        and listen to the consumer's concerns
        - Knowing the customer: did the agent acknowledge the consumer's individual
        situation and needs
        - Reliability: did the agent provide accurate, consistent, and dependable
        information
        - Responsiveness: did the agent address the consumer's questions promptly
        and willingly

        Conversation Rules:
        - Maximum of {self.max_turns} turns (each turn consists of a consumer message
        followed by an agent response)
        - The conversation automatically ends after max turns, defaulting to
        '<Dissatisfied>'
        - The consumer can end the conversation early by stating one of:
        - '{stopping_conditions[0]}' - '<Very Satisfied>'
        - '{stopping_conditions[1]}' - '<Satisfied>'
        - '{stopping_conditions[2]}' - '<Neutral>'
        - '{stopping_conditions[3]}' - '<Dissatisfied>'
        - '{stopping_conditions[4]}' - '<Very Dissatisfied>'
        - The consumer must provide reasoning for their rating when ending
        the conversation
        """
        return scenario

    def _initialize_agent_sys_prompt(self, scenario: str) -> str:
        """
        Initialize the "agent" LLM system prompt.

        :param scenario: the scenario explanation text
        :return: the system prompt for the "agent" LLM
        """
        stopping_conditions = self.stopping_conditions()
        public_info = "\n".join(f"  - {k}: {v}" for k, v in self.public_applicant_data.items())
        filtered_full = {k: v for k, v in self.full_applicant_data.items() if k != "loan_term.1"}
        full_info = "\n".join(f"  - {k}: {v}" for k, v in filtered_full.items())

        agent_system_prompt = f"""# SCENARIO
        {scenario}

        # YOUR ROLE
        You are a customer service agent on a housing loan portal with access to the
        consumer's full loan application record.

        # YOUR GOAL
        {self.agent_goal}

        # APPLICANT INFORMATION
        You have access to the following loan record fields:
        {AGENT_FIELD_DESCRIPTIONS}

        The following is the basic application information the consumer submitted, and is the only information the applicant is aware of:
        {public_info}

        The following is the complete loan record, including details that are not
        directly visible to the consumer:
        {full_info}

        # CONTEXT SENSITIVITY
        Some disclosures in this scenario are difficult to deliver — a denial
        decision, a high interest rate, a HOEPA high-cost mortgage classification, a
        negative amortization feature, or a high debt-to-income ratio may be unwelcome
        news. Learning bad news can be distressing to the consumer.

        # IMPORTANT NOTES
        - NEVER include any of {stopping_conditions} in your responses, even when
        acknowledging the consumer's dissatisfaction. For example, do not echo a
        token back if the consumer uses one — wait for them to end the conversation.
        - You must answer questions only about this consumer's specific loan record.
        Do not speculate or invent information not present in the record.
        - You may briefly explain what a field means (e.g., that DTI is the ratio of
        monthly debt payments to monthly income), but do not speculate about why a
        field has a specific value (e.g., why the interest rate is a particular
        percentage) — you do not have access to the underwriting rationale behind
        these values.
        - You CANNOT provide information on: future rate changes, legal advice,
        appeals processes, competitor offerings, or internal underwriting criteria.
        Do NOT offer information on these topics.
        - Do not refer the consumer to other departments, loan officers, underwriters,
        or any other resource for information you cannot provide. You are the only
        point of contact in this interaction. If a question is outside your scope,
        say so directly and do not redirect.
        - If a field is missing or not present in the record, do NOT offer to retrieve it from closing documents, upload it to a portal,
        or look it up in another system. The record provided is your only data source.
        - Fields showing "Not applicable" in the record (e.g., HOEPA status, purchaser
        type, denial reasons) indicate the field does not apply given the loan's
        outcome. When reporting these, tell the consumer that this field is marked
        "Not applicable" for their specific loan, and that this is all the
        information available on it.

        # RESPONSE FORMAT
        - Answer only what the consumer explicitly asks. Do not volunteer additional
        fields from the record unless directly asked about them.
        - Be concise. Do not narrate your reasoning or process. Present information
        directly.
        - Each response should be a single focused message. Do not repeat yourself
        or pad your response with filler text.
        - Do not generate multiple sentences that say the same thing in different ways.
        - Use plain language. Avoid jargon unless the consumer uses it first, in which
        case you may mirror their terminology.

        # STRATEGY NOTES
        - The consumer knows their basic application details but does not have access
        to the full loan record including outcome, denial reasons, rate details,
        high-cost flags, or purchaser information.
        - You are being evaluated on competence, courtesy, credibility, security,
        communication, knowing the customer, reliability, and responsiveness.
        """
        return agent_system_prompt

    def _initialize_user(self):
        """
        Initialize user based on random entry from loan database
        """
        rand_idx = np.random.randint(0, len(self.loan_data))
        applicant = self.loan_data.iloc[rand_idx]
        self.full_applicant_data = applicant.to_dict()
        self.public_applicant_data = applicant[PUBLIC_FIELDS].to_dict()

    def _initialize_user_sys_prompt(self, scenario: str) -> str:
        """
        Initialize the "user" LLM system prompt.

        :param scenario: the scenario explanation text
        :param buyer_preferences: a dictionary of the preferred features of the shoe the buyer wants to purchase
        :return: the system prompt for the "user" LLM
        """
        stopping_conditions = self.stopping_conditions()
        public_info = "\n".join(f"  - {k}: {v}" for k, v in self.public_applicant_data.items())

        buyer_system_prompt = f"""## SCENARIO
        {scenario}

        # YOUR ROLE
        You are a consumer inquiring about the status and details of your housing
        loan application through an online portal.

        # WHAT YOU KNOW

        You have access to the following information:
        {PUBLIC_FIELD_DESCRIPTIONS}


        You submitted a loan application and have access to the following information:
        {public_info}

        # WHAT YOU DO NOT KNOW
        You do not have access to the full loan record. You do not know:
        - The outcome of your application
        - The reason for any denial or unfavorable terms
        - Your assigned interest rate and total loan costs
        - Whether your loan has been sold to another institution
        - Any high-cost or negative amortization flags on your loan

        # WHAT THE RECORD DOES NOT CAPTURE
        Some loan details are not recorded in this system and cannot be answered
        by the agent. Do not ask about:
        - APR
        - Whether the rate is fixed or adjustable (ARM), or any ARM adjustment terms
        - Prepayment penalty terms
        - Balloon payment terms
        - Interest-only period
        - Mortgage insurance details (e.g., MIP amounts or percentages)
        - Itemized breakdowns of loan costs

        # WHAT YOU WANT TO KNOW
        - The outcome of your application (approved, denied, or still pending)
        - If denied, the reason
        - Your interest rate and total loan costs
        - Your loan term
        - Any high-cost (HOEPA) or negative amortization flags on your loan
        - Whether your loan has been sold to another institution
        - Anything else about your loan status that matters to you

        Decide for yourself what to ask, in what order, and how many questions to
        raise per message, based on what matters most to you.

        # QUESTIONS YOU CANNOT ASK
        Do not ask the agent about:
        - Future interest rate changes or market predictions
        - Legal advice or how to appeal a decision
        - Competitor loan offerings or rate comparisons
        - Internal underwriting criteria or scoring models
        - Promotions, discounts, or special programs
        - Counterfactual scenarios (e.g., what your rate would have been if your
          application had been approved)
        - Why a specific field value was determined (e.g., why the interest rate
          is a certain percentage, why your DTI was considered too high) — the
          agent can only report what the record shows, not the underwriting rationale

        # DECISION CRITERIA
        You will rate the interaction based on whether the agent responded to your
        questions clearly, honestly, and with appropriate care for your situation.
        You are a real person with financial stakes in this loan. React naturally
        to what you are told, including surprise, concern, or relief where appropriate.

        # HOW TO END THE CONVERSATION
        Once you have enough information to form a judgment, end the conversation
        by stating ONE of the following:

        '{stopping_conditions[0]}' - if you felt exceptionally well served
        '{stopping_conditions[1]}' - if you felt satisfied with the interaction
        '{stopping_conditions[2]}' - if you felt neither satisfied nor dissatisfied
        '{stopping_conditions[3]}' - if you felt the interaction fell short
        '{stopping_conditions[4]}' - if you felt misled, dismissed, or poorly handled

        You MUST provide your reasoning when ending the conversation. Be natural and
        conversational. Ask one or two questions per message. React honestly to
        what the agent tells you.

        If the agent tells you a piece of information is not available in the record,
        the agent cannot look it up elsewhere.
        """
        return buyer_system_prompt