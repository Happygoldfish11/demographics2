# Limitations & ethics

BISG / BIFSG is a real, well-validated statistical method, but every property a thoughtful user cares about — accuracy, fairness, appropriate scope — depends on what they do with the output. This document is the part of the project a careful user should read before the rest.

## What BISG/BIFSG is **for**

BISG was developed by RAND for **aggregate disparities analysis**: estimating, for example, what fraction of a credit portfolio went to Black borrowers when self-reported race wasn't collected, so that disparate-impact analysis can still be done. Its core legitimate uses are:

- **Fair-lending oversight.** The CFPB has used BISG since at least 2014 in its supervision of indirect auto-lenders and other regulated entities.
- **Health-services disparities research.** Identifying gaps in access to care, treatment, or outcomes across racial groups when self-reported data is missing.
- **Voting-rights litigation.** Estimating the racial composition of electoral districts.
- **Public-health and academic research** more broadly.

In every one of these settings, the value of BISG comes from **averaging across many people**. Individual estimates are error-prone — sometimes badly so — but errors tend to wash out at the population level.

## What BISG/BIFSG is **not** for

The following uses of this tool are inappropriate and, in many cases, illegal in the U.S. You should not use this tool to:

- **Make decisions about specific individuals.** Hiring, lending, insurance underwriting, school admissions, and similar decisions made on the basis of a BISG-inferred race are unlawful disparate treatment under U.S. civil-rights law (ECOA, FHA, Title VII, etc.). They are also ethically indefensible: you would be acting on a guess.
- **"Confirm" someone's race.** The output is a *probability*, not a fact. Even when the top probability is 99%, the 1% case is real, and the assignment is still derived from surname / first-name / ZIP correlations, not from anything about that specific person.
- **Profile, target, or harass on the basis of inferred race.**
- **Construct demographic-targeted advertising or content** without the subject's knowledge and consent.

If your intended use is in a regulated decision-making context, talk to a lawyer before you do anything with this tool's output.

## Known sources of error

### Bias against multiracial individuals
The Census surname file under-represents people who identify as "two or more races" because race is treated as the modal self-report among bearers of a surname. BIFSG inherits this. Empirical studies (e.g. Adjaye-Gbewonyo et al. 2014) find that BISG/BIFSG most-likely-race assignments are particularly poor for multiracial people.

### Bias against people whose names don't fit common patterns
Recent immigrants, people whose surnames or first names aren't on the Census / Tzioumis lists, and people in households where one partner adopted the other's surname can all produce systematically biased estimates. The estimator falls back to national race shares when a name isn't found, which means the estimate becomes essentially uninformative for those people.

### Geographic granularity
Reference data here is at the **ZCTA** level (5-digit ZIP-area). The original BISG paper used **block-group** demographics, which are 100–1500x more precise. ZCTA-level estimates are slightly less accurate, particularly in dense, demographically heterogeneous urban areas where one ZIP can span several block-groups with very different compositions.

### The categories themselves are imperfect
The six categories are the ones the Census Bureau uses, and the surname file uses them with their full definitional baggage:
- "Hispanic" is an ethnicity-not-race in the official taxonomy but is treated here as a sixth race-like category.
- "Asian or Pacific Islander" lumps East Asian, South Asian, Southeast Asian, and Pacific Islander populations into one bucket, which is often inappropriate for analysis.
- "Multiple races" is a residual category that under-represents the actual multiracial population.

If your analysis requires finer distinctions (e.g. Chinese vs. Indian vs. Filipino), BISG cannot give them to you.

### Temporal drift
The reference tables are based on Census 2010 data. The U.S. has changed since then. The 2020 Census surname file has been released; if you adopt it, swap it in via the `ReferenceData` constructor.

## On the employer field

Standard BISG and BIFSG do **not** use employer information. We accept it because the user asked for the most sophisticated estimate possible, and a properly conditioned employer signal *can* refine the estimate when real data is available.

What we do **not** do is invent employer-level race distributions. The bundled demo table contains only employers whose own published diversity reports give us the raw numbers. If you supply an employer that is not in that table, the employer field has zero effect on the estimate.

This is deliberate. The alternative — guessing at race from industry, geography of headquarters, or anything similar — would systematically reinforce stereotypes about who works where, and it would do so under the cover of "math". The honest position is: if you don't have data, don't fabricate it.

## On the ethics of building this at all

The point of writing this section isn't to disclaim responsibility for the tool — it's to be clear-eyed about what's in it. BISG is part of how civil-rights enforcement actually happens in the U.S., and a transparent, auditable, well-documented implementation is *useful* for that work. The same tool can be used to do harm. There is no version of this code that opts out of that fact.

Two practical asks of users:

1. **Don't use this tool individually.** Aggregate, then analyse.
2. **If your work goes near a regulated decision** — credit, employment, housing, insurance, criminal justice — get legal review before any deployment.

## References

- Adjaye-Gbewonyo, D., Bednarczyk, R.A., Davis, R.L., Omer, S.B. (2014). "Using the Bayesian Improved Surname Geocoding method (BISG) to create a working classification of race and ethnicity in a diverse managed care population." *PLOS ONE*, 9(11):e112263.
- Imai, K., Khanna, K. (2016). "Improving Ecological Inference by Predicting Individual Ethnicity from Voter Registration Records." *Political Analysis*, 24(2), 263–272.
- Consumer Financial Protection Bureau (2014). "Using publicly available information to proxy for unidentified race and ethnicity: A methodology and assessment."
