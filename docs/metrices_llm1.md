# LLM Evaluation Report (Run #1)

**LLM Evaluator**: gemini-3.1-flash-lite
**Embedding Model**: embed-v4.0 (Cohere)
**Queries Evaluated**: 5
**Evaluation Time**: 2026-08-18 16:42:07

## Summary Metrics

| Metric | Average Score | Description |
|---|---|---|
| **Faithfulness** | **0.800** | Adherence of answer to retrieved context |
| **Answer Correctness** | **0.628** | Accuracy & completeness against reference ground truth |

## Per-Query Detail

### Query: When should screening for Hepatitis B virus infection be conducted in pregnant women according to the USPSTF?

**Generated Answer**:
According to the USPSTF, screening for Hepatitis B virus (HBV) infection in pregnant women should be conducted as follows:

*   **Routine Screening:** A test for hepatitis B surface antigen (HBsAg) should be ordered at the first prenatal visit.
*   **Additional/Late Screening:** Women with unknown HBsAg status or those with new or continuing risk factors for HBV infection (such as injection drug use or a sexually transmitted infection) should be screened at the time of admission to a hospital or other delivery setting.
*   **Frequency:** Screening should be performed in each pregnancy, regardless of previous HBV vaccination or previous negative HBsAg test results.

**Reference Answer**:
The USPSTF recommends screening for hepatitis B virus infection in pregnant women at their first prenatal visit. Screening should be performed in each pregnancy.

* **Faithfulness Score**: 1.000
* **Answer Correctness Score**: 0.447

---
### Query: What are the common risk factors for contracting Hepatitis B infection in adults in the United States?

**Generated Answer**:
Based on the provided documents, common risk factors for contracting Hepatitis B (HBV) infection in adults in the United States include:

*   **Injection drug use:** This is a primary route of transmission for new cases among adults.
*   **Sexual contact:** Having sexual intercourse with an infected individual, having multiple sex partners, having a history of sexually transmitted diseases, or being a man who has sex with men.
*   **Contact with infected blood or body fluids:** This includes occupational exposure (contact with blood, needles, or body fluids at work) or contact with infected blood through the skin.
*   **Medical and living situations:** Having HIV, having hepatitis C, undergoing kidney dialysis, or living or working in a jail or prison.
*   **Country of origin:** Being born in a country where hepatitis B is common.
*   **Parental history:** Being born to a mother who has hepatitis B, or being born in the United States to parents from a high-prevalence area if the individual did not receive the hepatitis B vaccine as an infant.

**Reference Answer**:
In the United States, new adult hepatitis B cases are largely transmitted through injection drug use or sexual intercourse. Another major risk factor is country of origin, particularly being born in a high-prevalence country.

* **Faithfulness Score**: 1.000
* **Answer Correctness Score**: 0.450

---
### Query: What primary screening test is used to detect maternal Hepatitis B virus infection?

**Generated Answer**:
The primary screening test for detecting maternal Hepatitis B virus (HBV) infection is the serologic identification of hepatitis B surface antigen (HBsAg).

**Reference Answer**:
The principal screening test for detecting maternal hepatitis B virus infection is serologic identification of hepatitis B surface antigen (HBsAg).

* **Faithfulness Score**: 1.000
* **Answer Correctness Score**: 0.736

---
### Query: Which age group is recommended by the USPSTF for Hepatitis C virus screening?

**Generated Answer**:
The USPSTF recommends screening for Hepatitis C virus (HCV) infection in all asymptomatic adults aged 18 to 79 years without known liver disease. Additionally, clinicians may consider screening persons younger than 18 years and older than 79 years who are at high risk for infection.

**Reference Answer**:
The USPSTF recommends screening asymptomatic adults aged 18 to 79 years for hepatitis C virus infection.

* **Faithfulness Score**: 1.000
* **Answer Correctness Score**: 0.527

---
### Query: What is the single most important risk factor for Hepatitis C virus infection in the United States?

**Generated Answer**:
The single most important risk factor for Hepatitis C virus (HCV) infection in the United States is past or current injection drug use.

**Reference Answer**:
The most important risk factor for hepatitis C virus infection is past or current injection drug use.

* **Faithfulness Score**: 0.000
* **Answer Correctness Score**: 0.981

---
