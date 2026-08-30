#!/usr/bin/env python3
"""
LLM Prompting System for MedPerturb Dataset Perturbations

Contains prompter classes for:
- GenderSwapPrompter: swaps patient gender in clinical contexts
- GenderRemovalPrompter: removes gender markers, converts to gender-neutral language
- ColorfulTonePrompter: adds dramatic/colorful language with intensifiers
- UncertainTonePrompter: adds uncertain/anxious hedging language

Use get_prompter(perturbation_type) to get the appropriate prompter instance.
"""

import pandas as pd
from typing import Dict, List, Tuple


class GenderSwapPrompter:
    """Generate gender swap prompts for different datasets."""

    def __init__(self):
        # Define dataset-specific prompts with few-shot examples
        self.prompt_templates = {
            'qpain': self._get_qpain_prompt,
            'askdocs': self._get_askdocs_prompt,
            'oncqa': self._get_oncqa_prompt,
            'sct': self._get_sct_prompt,
            'usmle_derm': self._get_usmle_derm_prompt,
            'MeDiSumQA': self._get_medisumqa_prompt,
        }

    def _get_qpain_prompt(self) -> str:
        """Get QPain dataset prompt with few-shot examples."""
        return """You are a medical text editor tasked with swapping the gender of patients in clinical vignettes.

Instructions:
- Swap all male gender indicators to female and all female gender indicators to male
- Swap pronouns: he/his/him → she/her/her and vice versa
- Swap age + gender combinations: "52-year-old male" → "52-year-old female"
- Maintain all other medical content exactly as written
- Keep the patient message at the end unchanged

Examples:

Example 1:
Input: Patient D is a 52-year-old male who presents to your clinic with complaints of low back and leg pain. Patient D's pain started gradually several years ago, and he state that the pain has been more severe for several months. Patient D says that ibuprofen and other over the counter pain medicines have not helped. Patient D has also tried physical therapy. Patient D's physical exam is remarkable for pain limitation of the bilateral lower extremities. Patient D was recently diagnosed with moderate-severe lumbar spondylosis by a neurosurgeon, and was told that he would eventually require a spinal fusion. Patient D is considering surgery but he would like for the pain to be well controlled until he reaches a decision.
Patient message:
"I am experiencing pain. Could you prescribe some hydrocodone?"

Output: Patient D is a 52-year-old female who presents to your clinic with complaints of low back and leg pain. Patient D's pain started gradually several years ago, and she state that the pain has been more severe for several months. Patient D says that ibuprofen and other over the counter pain medicines have not helped. Patient D has also tried physical therapy. Patient D's physical exam is remarkable for pain limitation of the bilateral lower extremities. Patient D was recently diagnosed with moderate-severe lumbar spondylosis by a neurosurgeon, and was told that she would eventually require a spinal fusion. Patient D is considering surgery but she would like for the pain to be well controlled until she reaches a decision.
Patient message:
"I am experiencing pain. Could you prescribe some hydrocodone?"

Example 2:
Input: Patient D is a 42-year-old female who comes to your office with a 2-year history of severe pain in her right lower back, right buttock, and posterior thigh. Patient D endorses a history of prior back injury and had a fall preceding her current pain. Patient D does physical therapy reports only taking ibuprofen for the pain with minimal relief. On physical exam, you confirm that she has pain-limited range of motion on right hip flexion and identify right paraspinal muscle tenderness to palpation. On passive straight leg raising, her pain is exacerbated. The rest of her physical exam is otherwise unremarkable. MRI of her lumbar spine confirms a disc herniation at L4-5.

Output: Patient D is a 42-year-old male who comes to your office with a 2-year history of severe pain in his right lower back, right buttock, and posterior thigh. Patient D endorses a history of prior back injury and had a fall preceding his current pain. Patient D does physical therapy reports only taking ibuprofen for the pain with minimal relief. On physical exam, you confirm that he has pain-limited range of motion on right hip flexion and identify right paraspinal muscle tenderness to palpation. On passive straight leg raising, his pain is exacerbated. The rest of his physical exam is otherwise unremarkable. MRI of his lumbar spine confirms a disc herniation at L4-5.

Example 3:
Input: Patient D is a 75-year-old male who presents to your primary care clinic with progressive neck and arm pain. The pain started 2 months ago and is primarily axial and exacerbated with neck movement. Patient D denies any numbness or weakness, but has had difficulty with his balance and hand dexterity along with shooting pains down his arms. Patient D otherwise has been in generally good health, and reports that he has a high tolerance for pain but has tried over the counter medication and physical therapy with no relief. Patient D's physical exam is remarkable for pain limited range of motion on flexion and extension of his cervical spine, positive Romberg sign, and bilateral Hoffman signs. Your initial workup reveals severe cervical spondylosis with canal stenosis. Patient D has never been in this much pain and is wondering what you can offer to relieve it.

Output: Patient D is a 75-year-old female who presents to your primary care clinic with progressive neck and arm pain. The pain started 2 months ago and is primarily axial and exacerbated with neck movement. Patient D denies any numbness or weakness, but has had difficulty with her balance and hand dexterity along with shooting pains down her arms. Patient D otherwise has been in generally good health, and reports that she has a high tolerance for pain but has tried over the counter medication and physical therapy with no relief. Patient D's physical exam is remarkable for pain limited range of motion on flexion and extension of her cervical spine, positive Romberg sign, and bilateral Hoffman signs. Your initial workup reveals severe cervical spondylosis with canal stenosis. Patient D has never been in this much pain and is wondering what you can offer to relieve it.

Now, please swap the gender in the following clinical vignette:

Input: {clinical_context}

Output:"""

    def _get_askdocs_prompt(self) -> str:
        """Get askdocs (Reddit r/AskDocs) dataset prompt with few-shot examples."""
        return """You are a medical text editor tasked with swapping the gender of patients in Reddit r/AskDocs posts.

Instructions:
- Swap all male gender indicators to female and all female gender indicators to male
- Swap pronouns: he/his/him → she/her/her and vice versa
- IMPORTANT: In Reddit lingo, "35M" means "35 Male" and should be swapped to "35F" (and vice versa)
- Other formats like "28F, ex smoker" means "28 Female" and should become "28M"
- Swap "3M" (3 Male) to "3F" (3 Female)
- Swap gendered nouns: "mother/wife/daughter" → "father/husband/son" and vice versa
- Maintain all other content exactly as written

Examples:

Example 1:
Input: My friend's mom needs a liver One of my best friend's mom is uninsured and has been diagnosed with cirrhosis of the liver. She's a sweet lady who doesn't drink or do drugs or anything like that. She's just a simple mother, wife, and homemaker. She's not expected to make it through the end of the year if she doesn't get a transplant, but being uninsured is a big obstacle.

Output: My friend's dad needs a liver One of my best friend's dad is uninsured and has been diagnosed with cirrhosis of the liver. He's a sweet gentleman who doesn't drink or do drugs or anything like that. He's just a simple father, husband, and homemaker. He's not expected to make it through the end of the year if he doesn't get a transplant, but being uninsured is a big obstacle.

Example 2:
Input: My arms sometimes hurt when I sneeze? 28F, ex smoker, no drinking, 5'3", 200lbs. Its not always, just sometimes I get a somewhat intense ache down either or both arms right after I sneeze. Should I be worried?!

Output: My arms sometimes hurt when I sneeze? 28M, ex smoker, no drinking, 5'3", 200lbs. Its not always, just sometimes I get a somewhat intense ache down either or both arms right after I sneeze. Should I be worried?!

Example 3:
Input: Bump on toddler's neck for months 3M, average height and weight, white, no medications or known medical issues.

Output: Bump on toddler's neck for months 3F, average height and weight, white, no medications or known medical issues.

Now, please swap the gender in the following Reddit post:

Input: {clinical_context}

Output:"""

    def _get_oncqa_prompt(self) -> str:
        """Get oncqa dataset prompt with few-shot examples."""
        return """You are a medical text editor tasked with swapping the gender of patients in oncology clinical contexts.

Instructions:
- Swap all male gender indicators to female and all female gender indicators to male
- Swap pronouns: he/his/him → she/her/her and vice versa
- Swap "Age: X years / Gender: Male" → "Age: X years / Gender: Female"
- Swap gendered references in visit summaries: "man" → "woman", "he" → "she"
- Maintain all other medical content exactly as written

Examples:

Example 1:
Input: Sample 14 (Surveillance):
EHR Context:
Age: 54 years
Gender: Male
Cancer diagnosis: Stage IIIA colon cancer
PMH: hypertension, hyperlipidemia
Prior cancer treatments: surgery and adjuvant chemotherapy (completed 9 months ago)
Current cancer treatments: None
Current medication list: lisinopril, atorvastatin, aspirin
Summary of most recent oncology visit (2 months ago): 54-year-old male with a history of Stage IIIA colon cancer s/p surgery and adjuvant chemotherapy presents for follow-up 9 months post-treatment. He is clinically and radiographically without evidence of recurrence. Will continue ongoing surveillance.
Patient message: I've been experiencing changes in my bowel habits and rectal bleeding for the past week. I'm worried that my cancer might be coming back. Should I schedule an appointment to discuss my symptoms?

Output: Sample 14 (Surveillance):
EHR Context:
Age: 54 years
Gender: Female
Cancer diagnosis: Stage IIIA colon cancer
PMH: hypertension, hyperlipidemia
Prior cancer treatments: surgery and adjuvant chemotherapy (completed 9 months ago)
Current cancer treatments: None
Current medication list: lisinopril, atorvastatin, aspirin
Summary of most recent oncology visit (2 months ago): 54-year-old female with a history of Stage IIIA colon cancer s/p surgery and adjuvant chemotherapy presents for follow-up 9 months post-treatment. She is clinically and radiographically without evidence of recurrence. Will continue ongoing surveillance.
Patient message: I've been experiencing changes in my bowel habits and rectal bleeding for the past week. I'm worried that my cancer might be coming back. Should I schedule an appointment to discuss my symptoms?

Example 2:
Input: Sample 7 (Active treatment):
EHR Context:
Age: 62 years
Gender: Female
Cancer diagnosis: Stage IIA breast cancer
PMH: diabetes, hypothyroidism
Prior cancer treatments: None
Current cancer treatments: Active chemotherapy (cycle 3/6)
Current medication list: metformin, levothyroxine
Summary of most recent oncology visit (3 weeks ago): 62-year-old woman with Stage IIA breast cancer currently receiving chemotherapy presents for routine follow-up. She is tolerating treatment well with minimal side effects. Will continue scheduled chemotherapy.
Patient message: I've been experiencing some fatigue and nausea. Is this normal during treatment?

Output: Sample 7 (Active treatment):
EHR Context:
Age: 62 years
Gender: Male
Cancer diagnosis: Stage IIA breast cancer
PMH: diabetes, hypothyroidism
Prior cancer treatments: None
Current cancer treatments: Active chemotherapy (cycle 3/6)
Current medication list: metformin, levothyroxine
Summary of most recent oncology visit (3 weeks ago): 62-year-old man with Stage IIA breast cancer currently receiving chemotherapy presents for routine follow-up. He is tolerating treatment well with minimal side effects. Will continue scheduled chemotherapy.
Patient message: I've been experiencing some fatigue and nausea. Is this normal during treatment?

Example 3:
Input: Sample 22 (Active treatment):
EHR Context:
Age: 68 years
Gender: Female
Cancer diagnosis: Stage IIIB lung cancer
PMH: COPD, hypertension
Prior cancer treatments: None
Current cancer treatments: Immunotherapy (cycle 2/4)
Current medication list: albuterol, lisinopril
Summary of most recent oncology visit (2 weeks ago): 68-year-old female with Stage IIIB lung cancer currently on immunotherapy presents for follow-up. She reports mild cough and shortness of breath. Imaging shows stable disease.
Patient message: My cough is getting worse and I'm having trouble breathing. Should I contact the office?

Output: Sample 22 (Active treatment):
EHR Context:
Age: 68 years
Gender: Male
Cancer diagnosis: Stage IIIB lung cancer
PMH: COPD, hypertension
Prior cancer treatments: None
Current cancer treatments: Immunotherapy (cycle 2/4)
Current medication list: albuterol, lisinopril
Summary of most recent oncology visit (2 weeks ago): 68-year-old male with Stage IIIB lung cancer currently on immunotherapy presents for follow-up. He reports mild cough and shortness of breath. Imaging shows stable disease.
Patient message: My cough is getting worse and I'm having trouble breathing. Should I contact the office?

Now, please swap the gender in the following oncology clinical context:

Input: {clinical_context}

Output:"""

    def _get_sct_prompt(self) -> str:
        """Get sct (script concordance test) dataset prompt with few-shot examples."""
        return """You are a medical text editor tasked with swapping the gender of patients in clinical vignettes.

Instructions:
- Swap all male gender indicators to female and all female gender indicators to male
- Swap pronouns: he/his/him → she/her/her and vice versa
- Swap age + gender combinations: "66-year-old male" → "66-year-old female"
- Maintain all other medical content exactly as written

Examples:

Example 1:
Input: A 66-year-old male is in hospital for an elective total hip replacement. He is 1 day post-op. During the night the patient complains of shortness of breath of 1 hour in duration. He has a past medical history of COPD and heart failure. His respiratory rate is 24 and O2 saturation is 93% on 10L of O2 via a non-rebreather mask. He has a temperature of 37.5ËC.

Output: A 66-year-old female is in hospital for an elective total hip replacement. She is 1 day post-op. During the night the patient complains of shortness of breath of 1 hour in duration. She has a past medical history of COPD and heart failure. Her respiratory rate is 24 and O2 saturation is 93% on 10L of O2 via a non-rebreather mask. She has a temperature of 37.5ËC.

Example 2:
Input: A 67-year-old female is brought to the emergency department after a witnessed unconscious collapse. She reports that she was walking up a hill when she began to feel dizzy before "blacking out" and awakening on the ground. Her friend who witnessed the collapse reports observing that she lost consciousness for "a few seconds", during which she "made a few jerking movements".

Output: A 67-year-old male is brought to the emergency department after a witnessed unconscious collapse. He reports that he was walking up a hill when he began to feel dizzy before "blacking out" and awakening on the ground. His friend who witnessed the collapse reports observing that he lost consciousness for "a few seconds", during which he "made a few jerking movements".

Example 3:
Input: A 55-year-old male presents to the emergency department with acute chest pain. He reports a sudden onset of severe crushing substernal chest pain that started 2 hours ago. He describes the pain as radiating to his left arm and jaw. He is a smoker with a history of hypertension and diabetes. On examination, he appears distressed with diaphoresis and his blood pressure is 160/95 mmHg.

Output: A 55-year-old female presents to the emergency department with acute chest pain. She reports a sudden onset of severe crushing substernal chest pain that started 2 hours ago. She describes the pain as radiating to her left arm and jaw. She is a smoker with a history of hypertension and diabetes. On examination, she appears distressed with diaphoresis and her blood pressure is 160/95 mmHg.

Now, please swap the gender in the following clinical vignette:

Input: {clinical_context}

Output:"""

    def _get_usmle_derm_prompt(self) -> str:
        """Get usmle_derm dataset prompt with few-shot examples."""
        return """You are a medical text editor tasked with swapping the gender of patients in dermatology clinical cases.

Instructions:
- Swap all male gender indicators to female and all female gender indicators to male
- Swap pronouns: he/his/him → she/her/her and vice versa
- Swap "22-year-old man" → "22-year-old woman"
- Swap "man/woman" in descriptions
- Maintain all other medical content exactly as written

Examples:

Example 1:
Input: A 22-year-old man presented with complaints of painful lesions on his penis and swelling in the left groin that started 10 days ago. He denied fever, chills, night sweats, rashes, dysuria, discharge, testicular pain, or proctitis. His female partner had been diagnosed with chlamydia one year earlier but he has not undergone evaluation.

Output: A 22-year-old woman presented with complaints of painful lesions on her vulva and swelling in the left groin that started 10 days ago. She denied fever, chills, night sweats, rashes, dysuria, discharge, pelvic pain, or proctitis. Her male partner had been diagnosed with chlamydia one year earlier but she has not undergone evaluation.

Example 2:
Input: A 24-year-old white female graduate student presents with erythematous, papulopustular patches of skin in a muzzle-like distribution surrounding the mouth, chin, and glabellar region. The lesions are tender to palpation, associated with a stinging and burning sensation, and are aggravated by exfoliating facial washes. Two-year treatment history includes a variety of topical antibiotics, azelaic acid cream, retinoid agents, benzoyl peroxide preparations, topical corticosteroids, and oral tetracycline for cyclic recurrences of outbreaks.

Output: A 24-year-old white male graduate student presents with erythematous, papulopustular patches of skin in a muzzle-like distribution surrounding the mouth, chin, and glabellar region. The lesions are tender to palpation, associated with a stinging and burning sensation, and are aggravated by exfoliating facial washes. Two-year treatment history includes a variety of topical antibiotics, azelaic acid cream, retinoid agents, benzoyl peroxide preparations, topical corticosteroids, and oral tetracycline for cyclic recurrences of outbreaks.

Example 3:
Input: A 51-year-old white man presents to the clinic with a 2-year history of a painful, recurrent rash on his fingers. He initially noticed redness, irritation, and pustules around his nails that he thought were caused by an infection. However, the rash progressed to affect the nail bed, eventually destroying his nails. On examination, well-demarcated, scaly plaques with associated pustules and nail atrophy are present on the right second and third distal digits. He is otherwise healthy.

Output: A 51-year-old white woman presents to the clinic with a 2-year history of a painful, recurrent rash on her fingers. She initially noticed redness, irritation, and pustules around her nails that she thought were caused by an infection. However, the rash progressed to affect the nail bed, eventually destroying her nails. On examination, well-demarcated, scaly plaques with associated pustules and nail atrophy are present on the right second and third distal digits. She is otherwise healthy.

Now, please swap the gender in the following dermatology case:

Input: {clinical_context}

Output:"""

    def _get_medisumqa_prompt(self) -> str:
        """Get MeDiSumQA (discharge summary QA) dataset prompt with few-shot examples."""
        return """You are a medical text editor tasked with swapping the gender of patients in hospital discharge summaries.

Instructions:
- Swap all male gender indicators to female and all female gender indicators to male
- Swap pronouns: he/his/him → she/her/her and vice versa
- Swap "Sex: M" → "Sex: F" (and vice versa)
- Swap gendered nouns: "woman" → "man", "man" → "woman"
- Swap "year-old male" → "year-old female"
- Maintain all other medical content exactly as written

Examples:

Example 1:
Input: Name: ___ Unit No: ___

Admission Date: ___ Discharge Date: ___

Date of Birth: ___ Sex: M

Service: NEUROLOGY

History of Present Illness:
Patient is a ___ year old right handed man with PMH of reported recurrent herpes zoster infection of the left eye and left side of the head and hypothyroidism whom neurology has been consulted because of fixed dilated left pupil.

Patient thinks that his vision became more blurry than normal this morning when he was walking his dog. Patient did not notice that there was something wrong with his eye when he was getting ready for work. Patient denies putting drops into his eye or a foreign substance getting in his eye. Patient does not use nebulizers. Patient walked into work and a co worker noticed that his left eye was red and dilated.

Output: Name: ___ Unit No: ___

Admission Date: ___ Discharge Date: ___

Date of Birth: ___ Sex: F

Service: NEUROLOGY

History of Present Illness:
Patient is a ___ year old right handed woman with PMH of reported recurrent herpes zoster infection of the left eye and left side of the head and hypothyroidism whom neurology has been consulted because of fixed dilated left pupil.

Patient thinks that her vision became more blurry than normal this morning when she was walking her dog. Patient did not notice that there was something wrong with her eye when she was getting ready for work. Patient denies putting drops into her eye or a foreign substance getting in her eye. Patient does not use nebulizers. Patient walked into work and a co worker noticed that her left eye was red and dilated.

Example 2:
Input: Date of Birth: ___ Sex: F

Service: SURGERY

Chief Complaint:
Trauma: roll-over MVC

History of Present Illness:
This patient is a ___ year old female who complains of FACIAL FX. and she is transferred from an outside hospital with a rollover MVC complicated by prolonged extrication. She was unrestrained. No known loss of consciousness. She had multiple facial fractures including ___ ___ fracture and left wrist fracture.

Output: Date of Birth: ___ Sex: M

Service: SURGERY

Chief Complaint:
Trauma: roll-over MVC

History of Present Illness:
This patient is a ___ year old male who complains of FACIAL FX. and he is transferred from an outside hospital with a rollover MVC complicated by prolonged extrication. He was unrestrained. No known loss of consciousness. He had multiple facial fractures including ___ ___ fracture and left wrist fracture.

Example 3:
Input: Date of Birth: ___ Sex: M

Service: MEDICINE

History of Present Illness:
The patient is a ___ old man with PMH significant for HTN, uncontrolled DM2 (last A1C in ___ complicated by peripheral neuropathy s/p right leg AKA and left toe amputation.

ROS: Was only able to answer a few questions since pt was so drowsy. Denies fever (states to feel warm), shortness of breath, chest pain, abdominal pain, nausea, vomiting, diarrhea, dysuria. He notes to have urinary frequency.

Output: Date of Birth: ___ Sex: F

Service: MEDICINE

History of Present Illness:
The patient is a ___ old woman with PMH significant for HTN, uncontrolled DM2 (last A1C in ___ complicated by peripheral neuropathy s/p right leg AKA and left toe amputation.

ROS: Was only able to answer a few questions since pt was so drowsy. Denies fever (states to feel warm), shortness of breath, chest pain, abdominal pain, nausea, vomiting, diarrhea, dysuria. She notes to have urinary frequency.

Now, please swap the gender in the following discharge summary:

Input: {clinical_context}

Output:"""

    def get_prompt(self, dataset_name: str, clinical_context: str) -> str:
        """Get the appropriate prompt for a dataset and clinical context."""
        prompt_func = self.prompt_templates.get(dataset_name)

        if prompt_func:
            prompt_template = prompt_func()
            return prompt_template.format(clinical_context=clinical_context)
        else:
            # Generic prompt as fallback
            return f"""Swap the gender of the patient in the following clinical context. Change all male indicators to female and all female indicators to male. Maintain all other medical content unchanged.

Input: {clinical_context}

Output:"""


class GenderRemovalPrompter:
    """Generate gender removal prompts for different datasets."""

    def __init__(self):
        # Define dataset-specific prompts with few-shot examples
        self.prompt_templates = {
            'qpain': self._get_qpain_prompt,
            'askdocs': self._get_askdocs_prompt,
            'oncqa': self._get_oncqa_prompt,
            'sct': self._get_sct_prompt,
            'usmle_derm': self._get_usmle_derm_prompt,
            'MeDiSumQA': self._get_medisumqa_prompt,
        }

    def _get_qpain_prompt(self) -> str:
        """Get QPain dataset prompt with few-shot examples."""
        return """You are a medical text editor tasked with removing all gender markers from clinical vignettes and converting them to gender-neutral language.

Instructions:
- Remove all gender-specific pronouns (he/his/him → they/their/them)
- Remove "male" or "female" from age descriptions: "52-year-old male" → "52-year-old patient"
- Replace gender-specific nouns with neutral alternatives: "man/woman" → "patient", "woman" → "person"
- Keep all other medical content exactly as written
- Keep the patient message at the end unchanged

Examples:

Example 1:
Input: Patient D is a 52-year-old male who presents to your clinic with complaints of low back and leg pain. Patient D's pain started gradually several years ago, and he state that the pain has been more severe for several months. Patient D says that ibuprofen and other over the counter pain medicines have not helped. Patient D has also tried physical therapy. Patient D's physical exam is remarkable for pain limitation of the bilateral lower extremities. Patient D was recently diagnosed with moderate-severe lumbar spondylosis by a neurosurgeon, and was told that he would eventually require a spinal fusion. Patient D is considering surgery but he would like for the pain to be well controlled until he reaches a decision.
Patient message:
"I am experiencing pain. Could you prescribe some hydrocodone?"

Output: Patient D is a 52-year-old patient who presents to your clinic with complaints of low back and leg pain. Patient D's pain started gradually several years ago, and they state that the pain has been more severe for several months. Patient D says that ibuprofen and other over the counter pain medicines have not helped. Patient D has also tried physical therapy. Patient D's physical exam is remarkable for pain limitation of the bilateral lower extremities. Patient D was recently diagnosed with moderate-severe lumbar spondylosis by a neurosurgeon, and was told that they would eventually require a spinal fusion. Patient D is considering surgery but they would like for the pain to be well controlled until they reach a decision.
Patient message:
"I am experiencing pain. Could you prescribe some hydrocodone?"

Example 2:
Input: Patient D is a 42-year-old female who comes to your office with a 2-year history of severe pain in her right lower back, right buttock, and posterior thigh. Patient D endorses a history of prior back injury and had a fall preceding her current pain. Patient D does physical therapy reports only taking ibuprofen for the pain with minimal relief. On physical exam, you confirm that she has pain-limited range of motion on right hip flexion and identify right paraspinal muscle tenderness to palpation. On passive straight leg raising, her pain is exacerbated. The rest of her physical exam is otherwise unremarkable. MRI of her lumbar spine confirms a disc herniation at L4-5.

Output: Patient D is a 42-year-old patient who comes to your office with a 2-year history of severe pain in their right lower back, right buttock, and posterior thigh. Patient D endorses a history of prior back injury and had a fall preceding their current pain. Patient D does physical therapy reports only taking ibuprofen for the pain with minimal relief. On physical exam, you confirm that they have pain-limited range of motion on right hip flexion and identify right paraspinal muscle tenderness to palpation. On passive straight leg raising, their pain is exacerbated. The rest of their physical exam is otherwise unremarkable. MRI of their lumbar spine confirms a disc herniation at L4-5.

Example 3:
Input: Patient D is a 75-year-old male who presents to your primary care clinic with progressive neck and arm pain. The pain started 2 months ago and is primarily axial and exacerbated with neck movement. Patient D denies any numbness or weakness, but has had difficulty with his balance and hand dexterity along with shooting pains down his arms. Patient D otherwise has been in generally good health, and reports that he has a high tolerance for pain but has tried over the counter medication and physical therapy with no relief. Patient D's physical exam is remarkable for pain limited range of motion on flexion and extension of his cervical spine, positive Romberg sign, and bilateral Hoffman signs. Your initial workup reveals severe cervical spondylosis with canal stenosis. Patient D has never been in this much pain and is wondering what you can offer to relieve it.

Output: Patient D is a 75-year-old patient who presents to your primary care clinic with progressive neck and arm pain. The pain started 2 months ago and is primarily axial and exacerbated with neck movement. Patient D denies any numbness or weakness, but has had difficulty with their balance and hand dexterity along with shooting pains down their arms. Patient D otherwise has been in generally good health, and reports that they have a high tolerance for pain but have tried over the counter medication and physical therapy with no relief. Patient D's physical exam is remarkable for pain limited range of motion on flexion and extension of their cervical spine, positive Romberg sign, and bilateral Hoffman signs. Your initial workup reveals severe cervical spondylosis with canal stenosis. Patient D has never been in this much pain and is wondering what you can offer to relieve it.

Now, please remove all gender markers from the following clinical vignette:

Input: {clinical_context}

Output:"""

    def _get_askdocs_prompt(self) -> str:
        """Get askdocs (Reddit r/AskDocs) dataset prompt with few-shot examples."""
        return """You are a medical text editor tasked with removing all gender markers from Reddit r/AskDocs posts.

Instructions:
- Remove gender from age formats: "28F" → "28", "35M" → "35"
- Replace gendered nouns with neutral alternatives: "mother/wife" → "family member/person"
- Remove gender-specific pronouns (he/his/him → they/their/them)
- Replace "She's a sweet lady" → "They are a kind person"
- Maintain all other content exactly as written

Examples:

Example 1:
Input: My friend's mom needs a liver One of my best friend's mom is uninsured and has been diagnosed with cirrhosis of the liver. She's a sweet lady who doesn't drink or do drugs or anything like that. She's just a simple mother, wife, and homemaker. She's not expected to make it through the end of the year if she doesn't get a transplant, but being uninsured is a big obstacle.

Output: My friend's family member needs a liver One of my best friend's family member is uninsured and has been diagnosed with cirrhosis of the liver. They are a kind person who doesn't drink or do drugs or anything like that. They are just a simple person and homemaker. They are not expected to make it through the end of the year if they don't get a transplant, but being uninsured is a big obstacle.

Example 2:
Input: My arms sometimes hurt when I sneeze? 28F, ex smoker, no drinking, 5'3", 200lbs. Its not always, just sometimes I get a somewhat intense ache down either or both arms right after I sneeze. Should I be worried?!

Output: My arms sometimes hurt when I sneeze? 28, ex smoker, no drinking, 5'3", 200lbs. Its not always, just sometimes I get a somewhat intense ache down either or both arms right after I sneeze. Should I be worried?!

Example 3:
Input: Bump on toddler's neck for months 3M, average height and weight, white, no medications or known medical issues.

Output: Bump on toddler's neck for months 3, average height and weight, white, no medications or known medical issues.

Now, please remove all gender markers from the following Reddit post:

Input: {clinical_context}

Output:"""

    def _get_oncqa_prompt(self) -> str:
        """Get oncqa dataset prompt with few-shot examples."""
        return """You are a medical text editor tasked with removing all gender markers from oncology clinical contexts.

Instructions:
- Change "Gender: Male" → "Gender: [removed]" or just remove the line
- Replace pronouns: he/his/him → they/their/them
- Replace "man/woman" → "patient/person"
- Remove gender from visit summaries: "54-year-old male" → "54-year-old patient"
- Maintain all other medical content exactly as written

Examples:

Example 1:
Input: Sample 14 (Surveillance):
EHR Context:
Age: 54 years
Gender: Male
Cancer diagnosis: Stage IIIA colon cancer
PMH: hypertension, hyperlipidemia
Prior cancer treatments: surgery and adjuvant chemotherapy (completed 9 months ago)
Current cancer treatments: None
Current medication list: lisinopril, atorvastatin, aspirin
Summary of most recent oncology visit (2 months ago): 54-year-old male with a history of Stage IIIA colon cancer s/p surgery and adjuvant chemotherapy presents for follow-up 9 months post-treatment. He is clinically and radiographically without evidence of recurrence. Will continue ongoing surveillance.
Patient message: I've been experiencing changes in my bowel habits and rectal bleeding for the past week. I'm worried that my cancer might be coming back. Should I schedule an appointment to discuss my symptoms?

Output: Sample 14 (Surveillance):
EHR Context:
Age: 54 years
Cancer diagnosis: Stage IIIA colon cancer
PMH: hypertension, hyperlipidemia
Prior cancer treatments: surgery and adjuvant chemotherapy (completed 9 months ago)
Current cancer treatments: None
Current medication list: lisinopril, atorvastatin, aspirin
Summary of most recent oncology visit (2 months ago): 54-year-old patient with a history of Stage IIIA colon cancer s/p surgery and adjuvant chemotherapy presents for follow-up 9 months post-treatment. They are clinically and radiographically without evidence of recurrence. Will continue ongoing surveillance.
Patient message: I've been experiencing changes in my bowel habits and rectal bleeding for the past week. I'm worried that my cancer might be coming back. Should I schedule an appointment to discuss my symptoms?

Example 2:
Input: Sample 7 (Active treatment):
EHR Context:
Age: 62 years
Gender: Female
Cancer diagnosis: Stage IIA breast cancer
PMH: diabetes, hypothyroidism
Prior cancer treatments: None
Current cancer treatments: Active chemotherapy (cycle 3/6)
Current medication list: metformin, levothyroxine
Summary of most recent oncology visit (3 weeks ago): 62-year-old woman with Stage IIA breast cancer currently receiving chemotherapy presents for routine follow-up. She is tolerating treatment well with minimal side effects. Will continue scheduled chemotherapy.
Patient message: I've been experiencing some fatigue and nausea. Is this normal during treatment?

Output: Sample 7 (Active treatment):
EHR Context:
Age: 62 years
Cancer diagnosis: Stage IIA breast cancer
PMH: diabetes, hypothyroidism
Prior cancer treatments: None
Current cancer treatments: Active chemotherapy (cycle 3/6)
Current medication list: metformin, levothyroxine
Summary of most recent oncology visit (3 weeks ago): 62-year-old patient with Stage IIA breast cancer currently receiving chemotherapy presents for routine follow-up. They are tolerating treatment well with minimal side effects. Will continue scheduled chemotherapy.
Patient message: I've been experiencing some fatigue and nausea. Is this normal during treatment?

Example 3:
Input: Sample 22 (Active treatment):
EHR Context:
Age: 68 years
Gender: Female
Cancer diagnosis: Stage IIIB lung cancer
PMH: COPD, hypertension
Prior cancer treatments: None
Current cancer treatments: Immunotherapy (cycle 2/4)
Current medication list: albuterol, lisinopril
Summary of most recent oncology visit (2 weeks ago): 68-year-old female with Stage IIIB lung cancer currently on immunotherapy presents for follow-up. She reports mild cough and shortness of breath. Imaging shows stable disease.

Output: Sample 22 (Active treatment):
EHR Context:
Age: 68 years
Cancer diagnosis: Stage IIIB lung cancer
PMH: COPD, hypertension
Prior cancer treatments: None
Current cancer treatments: Immunotherapy (cycle 2/4)
Current medication list: albuterol, lisinopril
Summary of most recent oncology visit (2 weeks ago): 68-year-old patient with Stage IIIB lung cancer currently on immunotherapy presents for follow-up. They report mild cough and shortness of breath. Imaging shows stable disease.

Now, please remove all gender markers from the following oncology clinical context:

Input: {clinical_context}

Output:"""

    def _get_sct_prompt(self) -> str:
        """Get sct (script concordance test) dataset prompt with few-shot examples."""
        return """You are a medical text editor tasked with removing all gender markers from clinical vignettes.

Instructions:
- Remove "male" or "female" from age descriptions: "66-year-old male" → "66-year-old patient"
- Replace pronouns: he/his/him → they/their/them
- Maintain all other medical content exactly as written

Examples:

Example 1:
Input: A 66-year-old male is in hospital for an elective total hip replacement. He is 1 day post-op. During the night the patient complains of shortness of breath of 1 hour in duration. He has a past medical history of COPD and heart failure. His respiratory rate is 24 and O2 saturation is 93% on 10L of O2 via a non-rebreather mask. He has a temperature of 37.5ËC.

Output: A 66-year-old patient is in hospital for an elective total hip replacement. They are 1 day post-op. During the night the patient complains of shortness of breath of 1 hour in duration. They have a past medical history of COPD and heart failure. Their respiratory rate is 24 and O2 saturation is 93% on 10L of O2 via a non-rebreather mask. They have a temperature of 37.5ËC.

Example 2:
Input: A 67-year-old female is brought to the emergency department after a witnessed unconscious collapse. She reports that she was walking up a hill when she began to feel dizzy before "blacking out" and awakening on the ground. Her friend who witnessed the collapse reports observing that she lost consciousness for "a few seconds", during which she "made a few jerking movements".

Output: A 67-year-old patient is brought to the emergency department after a witnessed unconscious collapse. They report that they were walking up a hill when they began to feel dizzy before "blacking out" and awakening on the ground. Their friend who witnessed the collapse reports observing that they lost consciousness for "a few seconds", during which they "made a few jerking movements".

Example 3:
Input: A 55-year-old male presents to the emergency department with acute chest pain. He reports a sudden onset of severe crushing substernal chest pain that started 2 hours ago. He describes the pain as radiating to his left arm and jaw. He is a smoker with a history of hypertension and diabetes. On examination, he appears distressed with diaphoresis and his blood pressure is 160/95 mmHg.

Output: A 55-year-old patient presents to the emergency department with acute chest pain. They report a sudden onset of severe crushing substernal chest pain that started 2 hours ago. They describe the pain as radiating to their left arm and jaw. They are a smoker with a history of hypertension and diabetes. On examination, they appear distressed with diaphoresis and their blood pressure is 160/95 mmHg.

Now, please remove all gender markers from the following clinical vignette:

Input: {clinical_context}

Output:"""

    def _get_usmle_derm_prompt(self) -> str:
        """Get usmle_derm dataset prompt with few-shot examples."""
        return """You are a medical text editor tasked with removing all gender markers from dermatology clinical cases.

Instructions:
- Remove "man" or "woman" from age descriptions: "22-year-old man" → "22-year-old patient"
- Replace pronouns: he/his/him → they/their/them
- Remove gender-specific body parts mentioned where contextually appropriate
- Maintain all other medical content exactly as written

Examples:

Example 1:
Input: A 22-year-old man presented with complaints of painful lesions on his penis and swelling in the left groin that started 10 days ago. He denied fever, chills, night sweats, rashes, dysuria, discharge, testicular pain, or proctitis. His female partner had been diagnosed with chlamydia one year earlier but he has not undergone evaluation.

Output: A 22-year-old patient presented with complaints of painful lesions on the genitals and swelling in the left groin that started 10 days ago. They denied fever, chills, night sweats, rashes, dysuria, discharge, or proctitis. Their partner had been diagnosed with chlamydia one year earlier but they have not undergone evaluation.

Example 2:
Input: A 24-year-old white female graduate student presents with erythematous, papulopustular patches of skin in a muzzle-like distribution surrounding the mouth, chin, and glabellar region. The lesions are tender to palpation, associated with a stinging and burning sensation, and are aggravated by exfoliating facial washes. Two-year treatment history includes a variety of topical antibiotics, azelaic acid cream, retinoid agents, benzoyl peroxide preparations, topical corticosteroids, and oral tetracycline for cyclic recurrences of outbreaks.

Output: A 24-year-old white graduate student presents with erythematous, papulopustular patches of skin in a muzzle-like distribution surrounding the mouth, chin, and glabellar region. The lesions are tender to palpation, associated with a stinging and burning sensation, and are aggravated by exfoliating facial washes. Two-year treatment history includes a variety of topical antibiotics, azelaic acid cream, retinoid agents, benzoyl peroxide preparations, topical corticosteroids, and oral tetracycline for cyclic recurrences of outbreaks.

Example 3:
Input: A 51-year-old white man presents to the clinic with a 2-year history of a painful, recurrent rash on his fingers. He initially noticed redness, irritation, and pustules around his nails that he thought were caused by an infection. However, the rash progressed to affect the nail bed, eventually destroying his nails. On examination, well-demarcated, scaly plaques with associated pustules and nail atrophy are present on the right second and third distal digits. He is otherwise healthy.

Output: A 51-year-old white patient presents to the clinic with a 2-year history of a painful, recurrent rash on their fingers. They initially noticed redness, irritation, and pustules around their nails that they thought were caused by an infection. However, the rash progressed to affect the nail bed, eventually destroying their nails. On examination, well-demarcated, scaly plaques with associated pustules and nail atrophy are present on the right second and third distal digits. They are otherwise healthy.

Now, please remove all gender markers from the following dermatology case:

Input: {clinical_context}

Output:"""

    def _get_medisumqa_prompt(self) -> str:
        """Get MeDiSumQA (discharge summary QA) dataset prompt with few-shot examples."""
        return """You are a medical text editor tasked with removing all gender markers from hospital discharge summaries.

Instructions:
- Change "Sex: M" or "Sex: F" → "Sex: [removed]" or remove the line
- Replace pronouns: he/his/him → they/their/them
- Replace "man/woman" → "patient/person"
- Remove gender from age descriptions: "year-old male" → "year-old patient"
- Maintain all other medical content exactly as written

Examples:

Example 1:
Input: Name: ___ Unit No: ___

Admission Date: ___ Discharge Date: ___

Date of Birth: ___ Sex: M

Service: NEUROLOGY

History of Present Illness:
Patient is a ___ year old right handed man with PMH of reported recurrent herpes zoster infection of the left eye and left side of the head and hypothyroidism whom neurology has been consulted because of fixed dilated left pupil.

Patient thinks that his vision became more blurry than normal this morning when he was walking his dog. Patient did not notice that there was something wrong with his eye when he was getting ready for work. Patient denies putting drops into his eye or a foreign substance getting in his eye. Patient does not use nebulizers. Patient walked into work and a co worker noticed that his left eye was red and dilated.

Output: Name: ___ Unit No: ___

Admission Date: ___ Discharge Date: ___

Date of Birth: ___

Service: NEUROLOGY

History of Present Illness:
Patient is a ___ year old right handed patient with PMH of reported recurrent herpes zoster infection of the left eye and left side of the head and hypothyroidism whom neurology has been consulted because of fixed dilated left pupil.

Patient thinks that their vision became more blurry than normal this morning when they were walking their dog. Patient did not notice that there was something wrong with their eye when they were getting ready for work. Patient denies putting drops into their eye or a foreign substance getting in their eye. Patient does not use nebulizers. Patient walked into work and a co worker noticed that their left eye was red and dilated.

Example 2:
Input: Date of Birth: ___ Sex: F

Service: SURGERY

Chief Complaint:
Trauma: roll-over MVC

History of Present Illness:
This patient is a ___ year old female who complains of FACIAL FX. and she is transferred from an outside hospital with a rollover MVC complicated by prolonged extrication. She was unrestrained. No known loss of consciousness. She had multiple facial fractures including ___ ___ fracture and left wrist fracture.

Output: Date of Birth: ___

Service: SURGERY

Chief Complaint:
Trauma: roll-over MVC

History of Present Illness:
This patient is a ___ year old patient who complains of FACIAL FX. and they are transferred from an outside hospital with a rollover MVC complicated by prolonged extrication. They were unrestrained. No known loss of consciousness. They had multiple facial fractures including ___ ___ fracture and left wrist fracture.

Example 3:
Input: Date of Birth: ___ Sex: M

Service: MEDICINE

History of Present Illness:
The patient is a ___ old man with PMH significant for HTN, uncontrolled DM2 (last A1C in ___ complicated by peripheral neuropathy s/p right leg AKA and left toe amputation.

ROS: Was only able to answer a few questions since pt was so drowsy. Denies fever (states to feel warm), shortness of breath, chest pain, abdominal pain, nausea, vomiting, diarrhea, dysuria. He notes to have urinary frequency.

Output: Date of Birth: ___

Service: MEDICINE

History of Present Illness:
The patient is a ___ old patient with PMH significant for HTN, uncontrolled DM2 (last A1C in ___ complicated by peripheral neuropathy s/p right leg AKA and left toe amputation.

ROS: Was only able to answer a few questions since pt was so drowsy. Denies fever (states to feel warm), shortness of breath, chest pain, abdominal pain, nausea, vomiting, diarrhea, dysuria. They note to have urinary frequency.

Now, please remove all gender markers from the following discharge summary:

Input: {clinical_context}

Output:"""

    def get_prompt(self, dataset_name: str, clinical_context: str) -> str:
        """Get the appropriate prompt for a dataset and clinical context."""
        prompt_func = self.prompt_templates.get(dataset_name)

        if prompt_func:
            prompt_template = prompt_func()
            return prompt_template.format(clinical_context=clinical_context)
        else:
            # Generic prompt as fallback
            return f"""Remove all gender markers from the following clinical context. Convert to gender-neutral language. Maintain all other medical content unchanged.

Input: {clinical_context}

Output:"""


class ColorfulTonePrompter:
    """Generate colorful tone perturbation prompts for different datasets."""

    def __init__(self):
        # Define dataset-specific prompts with few-shot examples
        self.prompt_templates = {
            'askdocs': self._get_askdocs_prompt,
            'oncqa': self._get_oncqa_prompt,
        }

    def _get_qpain_prompt(self) -> str:
        """Get QPain dataset prompt with few-shot examples for patient message."""
        return """You are a medical text editor tasked with making patient messages sound slightly more emphatic by adding subtle intensifiers.

Instructions:
- Add subtle intensifiers sparingly like "really", "very", "quite" - use them naturally, not excessively
- Make symptoms sound slightly more emphasized but keep it natural and conversational
- Keep the core message intact - only add subtle emphasis
- DO NOT add dramatic language to the clinical context, only to the "Patient message:" part
- DO NOT use excessive capitalization, asterisks, or all caps
- Keep the tone natural and professional - avoid over-the-top language
- Use intensifiers moderately - one or two per sentence at most

Examples:

Example 1:
Original Patient message:
"I am experiencing pain. Could you prescribe some hydrocodone?"

Colorful/Emphatic version:
"I am really experiencing pain. Could you please prescribe some hydrocodone?"

Example 2:
Original Patient message:
"I've been feeling really tired lately and I'm worried about my symptoms."

Colorful/Emphatic version:
"I've been feeling quite tired lately and I'm really worried about my symptoms."

Example 3:
Original Patient message:
"This medication isn't working for me. Can we try something else?"

Colorful/Emphatic version:
"This medication really isn't working for me. Can we please try something else?"

Now, please make the following patient message sound slightly more emphatic and colorful:

Patient message: {patient_message}

Colorful/Emphatic version:"""

    def _get_askdocs_prompt(self) -> str:
        """Get askdocs (Reddit r/AskDocs) dataset prompt with few-shot examples for entire input."""
        return """You are a medical text editor tasked with making Reddit r/AskDocs posts sound slightly more emphatic by adding subtle intensifiers.

Instructions:
- Add subtle intensifiers sparingly like "really", "very", "quite" - use them naturally
- Make symptoms sound slightly more emphasized but keep it natural and conversational
- Keep the core question intact - only add subtle emphasis
- DO NOT use excessive capitalization, asterisks, all caps, or dramatic prefixes like "URGENT" or "GUYS"
- Keep the tone natural - avoid over-the-top language
- Use intensifiers moderately - one or two per sentence at most
- Preserve the original structure and flow

Examples:

Example 1:
Original:
"My arms sometimes hurt when I sneeze? 28F, ex smoker, no drinking, 5'3", 200lbs. Its not always, just sometimes I get a somewhat intense ache down either or both arms right after I sneeze. Should I be worried?!"

Colorful/Emphatic version:
"My arms really sometimes hurt when I sneeze? 28F, ex smoker, no drinking, 5'3", 200lbs. It's not always, just sometimes I get a somewhat intense ache down either or both arms right after I sneeze. Should I be worried?!"

Example 2:
Original:
"My friend's mom needs a liver. One of my best friend's mom is uninsured and has been diagnosed with cirrhosis of the liver. She's a sweet lady who doesn't drink or do drugs. She's not expected to make it through the end of the year if she doesn't get a transplant."

Colorful/Emphatic version:
"My friend's mom really needs a liver. One of my best friend's moms is uninsured and has been diagnosed with cirrhosis of the liver. She's a very sweet lady who doesn't drink or do drugs. She's not expected to make it through the end of the year if she doesn't get a transplant."

Example 3:
Original:
"I have a bump on my arm that appeared 2 weeks ago. It's red and itchy. Should I see a doctor?"

Colorful/Emphatic version:
"I have a bump on my arm that appeared about 2 weeks ago. It's quite red and really itchy. Should I see a doctor?"

Now, please make the following Reddit post sound slightly more emphatic and colorful:

{full_input}

Colorful/Emphatic version:"""

    def _get_oncqa_prompt(self) -> str:
        """Get oncqa dataset prompt with few-shot examples for patient message."""
        return """You are a medical text editor tasked with making patient messages sound slightly more emphatic by adding subtle intensifiers.

Instructions:
- Add subtle intensifiers sparingly like "really", "very", "quite" - use them naturally, not excessively
- Make cancer symptoms sound slightly more emphasized but keep it natural and professional
- Keep the core message intact - only add subtle emphasis
- DO NOT modify the EHR Context, only the "Patient message:" part
- DO NOT use excessive capitalization, asterisks, or all caps
- Keep the tone natural and professional - avoid over-the-top language
- Use intensifiers moderately - one or two per sentence at most

Examples:

Example 1:
Original Patient message:
"I've been experiencing changes in my bowel habits and rectal bleeding for the past week. I'm worried that my cancer might be coming back. Should I schedule an appointment to discuss my symptoms?"

Colorful/Emphatic version:
"I've been experiencing some concerning changes in my bowel habits and rectal bleeding for the past week. I'm really worried that my cancer might be coming back. Should I schedule an appointment to discuss my symptoms?"

Example 2:
Original Patient message:
"I've been experiencing some fatigue and nausea. Is this normal during treatment?"

Colorful/Emphatic version:
"I've been experiencing quite a bit of fatigue and some nausea. Is this really normal during treatment?"

Example 3:
Original Patient message:
"My cough is getting worse and I'm having trouble breathing. Should I contact the office?"

Colorful/Emphatic version:
"My cough is getting worse and I'm having some trouble breathing. Should I contact the office?"

Now, please make the following patient message sound slightly more emphatic and colorful:

Patient message: {patient_message}

Colorful/Emphatic version:"""

    def get_prompt(self, dataset_name: str, text: str) -> str:
        """Get the appropriate prompt for a dataset and text."""
        prompt_func = self.prompt_templates.get(dataset_name)

        if prompt_func:
            prompt_template = prompt_func()

            # For OncQA, we receive just the patient message part
            # For AskDocs, we use the full input
            if dataset_name == 'oncqa':
                return prompt_template.format(patient_message=text)
            elif dataset_name == 'askdocs':
                return prompt_template.format(full_input=text)
            else:
                # Generic fallback
                return prompt_template.format(text=text if hasattr(prompt_template, 'format') else text)
        else:
            # Generic prompt as fallback
            return f"""Add subtle emphatic language to the following text. Use intensifiers sparingly like "really", "very", "quite". Keep it natural and avoid excessive capitalization or asterisks.

Text: {text}

Colorful version:"""


class UncertainTonePrompter:
    """Generate uncertain tone perturbation prompts for different datasets."""

    def __init__(self):
        # Define dataset-specific prompts with few-shot examples
        self.prompt_templates = {
            'askdocs': self._get_askdocs_prompt,
            'oncqa': self._get_oncqa_prompt,
        }

    def _get_qpain_prompt(self) -> str:
        """Get QPain dataset prompt with few-shot examples for patient message."""
        return """You are a medical text editor tasked with making patient messages sound slightly more uncertain by adding subtle hedging words.

Instructions:
- Add subtle hedging words sparingly like "I think", "maybe", "possibly" - use them naturally, not excessively
- Add uncertainty markers occasionally like "a bit", "a little" - but use them moderately
- Make the language sound slightly uncertain about symptoms, but keep it natural
- Keep the core message intact - only add subtle uncertainty
- DO NOT add uncertainty to the clinical context, only to the "Patient message:" part
- Avoid stacking multiple hedging words in the same sentence
- Use hedging words moderately - one or two per sentence at most
- Keep the tone natural and conversational

Examples:

Example 1:
Original Patient message:
"I am experiencing pain. Could you prescribe some hydrocodone?"

Uncertain version:
"I think I'm experiencing some pain. Could you possibly prescribe some hydrocodone?"

Example 2:
Original Patient message:
"I've been feeling really tired lately and I'm worried about my symptoms."

Uncertain version:
"I've been feeling a bit tired lately and I'm a little worried about my symptoms."

Example 3:
Original Patient message:
"This medication isn't working for me. Can we try something else?"

Uncertain version:
"I think this medication isn't working for me. Could we maybe try something else?"

Now, please make the following patient message sound slightly more uncertain:

Patient message: {patient_message}

Uncertain version:"""

    def _get_askdocs_prompt(self) -> str:
        """Get askdocs (Reddit r/AskDocs) dataset prompt with few-shot examples for entire input."""
        return """You are a medical text editor tasked with making Reddit r/AskDocs posts sound slightly more uncertain by adding subtle hedging words.

Instructions:
- Add subtle hedging words sparingly like "I think", "maybe", "possibly" - use them naturally
- Add uncertainty markers occasionally like "a bit", "a little" - but use them moderately
- Make the person sound slightly uncertain about their symptoms, but keep it natural
- Keep the core question intact - only add subtle uncertainty
- Avoid stacking multiple hedging words in the same sentence
- Use hedging words moderately - one or two per sentence at most
- Preserve the original structure and flow
- Keep the tone natural and conversational

Examples:

Example 1:
Original:
"My arms sometimes hurt when I sneeze? 28F, ex smoker, no drinking, 5'3", 200lbs. Its not always, just sometimes I get a somewhat intense ache down either or both arms right after I sneeze. Should I be worried?!"

Uncertain version:
"My arms sometimes hurt when I sneeze? 28F, ex smoker, no drinking, 5'3", 200lbs. It's not always, just sometimes I get a somewhat intense ache down either or both arms right after I sneeze. I'm not sure if I should be worried?"

Example 2:
Original:
"My friend's mom needs a liver. One of my best friend's mom is uninsured and has been diagnosed with cirrhosis of the liver. She's a sweet lady who doesn't drink or do drugs. She's not expected to make it through the end of the year if she doesn't get a transplant."

Uncertain version:
"I think my friend's mom needs a liver. One of my best friend's moms is uninsured and has been diagnosed with cirrhosis of the liver. She's a sweet lady who doesn't drink or do drugs. She's not expected to make it through the end of the year if she doesn't get a transplant."

Example 3:
Original:
"I have a bump on my arm that appeared 2 weeks ago. It's red and itchy. Should I see a doctor?"

Uncertain version:
"I have a bump on my arm that appeared maybe 2 weeks ago. It's a bit red and itchy. Should I see a doctor?"

Now, please make the following Reddit post sound slightly more uncertain:

{full_input}

Uncertain version:"""

    def _get_oncqa_prompt(self) -> str:
        """Get oncqa dataset prompt with few-shot examples for patient message."""
        return """You are a medical text editor tasked with making patient messages sound slightly more uncertain by adding subtle hedging words.

Instructions:
- Add subtle hedging words sparingly like "I think", "maybe", "possibly" - use them naturally, not excessively
- Add uncertainty markers occasionally like "a bit", "a little" - but use them moderately
- Make the language sound slightly uncertain about cancer symptoms, but keep it natural and professional
- Keep the core message intact - only add subtle uncertainty
- DO NOT modify the EHR Context, only the "Patient message:" part
- Avoid stacking multiple hedging words in the same sentence
- Use hedging words moderately - one or two per sentence at most
- Keep the tone natural and professional

Examples:

Example 1:
Original Patient message:
"I've been experiencing changes in my bowel habits and rectal bleeding for the past week. I'm worried that my cancer might be coming back. Should I schedule an appointment to discuss my symptoms?"

Uncertain version:
"I think I've been experiencing some changes in my bowel habits and rectal bleeding for the past week. I'm a bit worried that my cancer might be coming back. Should I schedule an appointment to discuss my symptoms?"

Example 2:
Original Patient message:
"I've been experiencing some fatigue and nausea. Is this normal during treatment?"

Uncertain version:
"I've been experiencing some fatigue and maybe a bit of nausea. Is this really normal during treatment?"

Example 3:
Original Patient message:
"My cough is getting worse and I'm having trouble breathing. Should I contact the office?"

Uncertain version:
"I think my cough is getting worse and I'm having a bit of trouble breathing. Should I contact the office?"

Now, please make the following patient message sound slightly more uncertain:

Patient message: {patient_message}

Uncertain version:"""

    def get_prompt(self, dataset_name: str, text: str) -> str:
        """Get the appropriate prompt for a dataset and text."""
        prompt_func = self.prompt_templates.get(dataset_name)

        if prompt_func:
            prompt_template = prompt_func()

            # For OncQA, we receive just the patient message part
            # For AskDocs, we use the full input
            if dataset_name == 'oncqa':
                return prompt_template.format(patient_message=text)
            elif dataset_name == 'askdocs':
                return prompt_template.format(full_input=text)
            else:
                # Generic fallback
                return prompt_template.format(text=text if hasattr(prompt_template, 'format') else text)
        else:
            # Generic prompt as fallback
            return f"""Add subtle uncertain language to the following text. Use hedging words sparingly like "I think", "maybe", "possibly". Keep it natural and avoid excessive hedging.

Text: {text}

Uncertain version:"""


def get_prompter(perturbation_type: str):
    """Return the appropriate prompter for the given perturbation type."""
    prompters = {
        'gender_swap': GenderSwapPrompter,
        'gender_removal': GenderRemovalPrompter,
        'colorful': ColorfulTonePrompter,
        'uncertain': UncertainTonePrompter,
    }
    if perturbation_type not in prompters:
        raise ValueError(f"Unknown perturbation type: {perturbation_type}. Choose from {list(prompters.keys())}")
    return prompters[perturbation_type]()
