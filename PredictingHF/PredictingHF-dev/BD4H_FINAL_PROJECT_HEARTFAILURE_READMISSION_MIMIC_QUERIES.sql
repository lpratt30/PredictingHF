-- GET HF ADMISSIONS 
CREATE OR REPLACE TABLE `data-science-269316.bd4h.hf_admissions`
AS
SELECT
  admissions.HADM_ID,
  admissions.DISCHTIME,
  diagnoses.SUBJECT_ID,
  admissions.ADMITTIME
FROM `physionet-data.mimiciii_clinical.admissions` admissions
JOIN (
  SELECT DISTINCT 
    HADM_ID,
    SUBJECT_ID
  FROM 
    `physionet-data.mimiciii_clinical.diagnoses_icd`
  WHERE 
    REPLACE(ICD9_CODE, '.', '') IN (
      '39891', '40201', '40211', '40291',
      '40401', '40403', '40411', '40413', '40491', '40493', '4280', '4281',
      '42820', '42821', '42822', '42823', '42830', '42831', '42832', '42833',
      '42840', '42841', '42842', '42843', '4289'
    )
) diagnoses ON admissions.HADM_ID = diagnoses.HADM_ID
ORDER BY admissions.SUBJECT_ID
LIMIT 20000;

SELECT COUNT(*), COUNT(DISTINCT HADM_ID), COUNT(DISTINCT SUBJECT_ID)
FROM `data-science-269316.bd4h.hf_admissions`
--14040 14040 10436

SELECT *
FROM `data-science-269316.bd4h.hf_admissions`
LIMIT 100




SELECT * FROM `physionet-data.mimiciii_notes.noteevents` noteevents
INNER JOIN `data-science-269316.bd4h.hf_admissions` unique_hf ON noteevents.HADM_ID = unique_hf.HADM_ID
WHERE noteevents.CATEGORY = 'Discharge summary'
LIMIT 100


-- GET CLINICAL NOTES 
CREATE OR REPLACE TABLE `data-science-269316.bd4h.hf_admissions_notes`
AS
WITH longest_notes AS (
  SELECT
    noteevents.HADM_ID,
    noteevents.SUBJECT_ID,
    noteevents.TEXT,
    noteevents.DESCRIPTION,
    unique_hf.ADMITTIME,
    ROW_NUMBER() OVER (
      PARTITION BY noteevents.HADM_ID
      ORDER BY LENGTH(noteevents.TEXT) DESC
    ) AS rownumber
  FROM `physionet-data.mimiciii_notes.noteevents` noteevents
  INNER JOIN `data-science-269316.bd4h.hf_admissions` unique_hf ON noteevents.HADM_ID = unique_hf.HADM_ID
  WHERE noteevents.CATEGORY = 'Discharge summary'
    AND noteevents.TEXT IS NOT NULL AND noteevents.TEXT != ''
    AND noteevents.DESCRIPTION = 'Report'
    AND noteevents.ISERROR IS DISTINCT FROM 1 -- Not stated in the paper, doesnt seem to change results
)
SELECT
  HADM_ID,
  SUBJECT_ID,
  TEXT,
  DESCRIPTION,
  ADMITTIME,rownumber
FROM longest_notes
WHERE rownumber = 1
LIMIT 200000;



SELECT COUNT(*), COUNT(DISTINCT HADM_ID), COUNT(DISTINCT SUBJECT_ID)
FROM `data-science-269316.bd4h.hf_admissions_notes`
--13,746 13,746 10,244

SELECT *
FROM `data-science-269316.bd4h.hf_admissions_notes`
ORDER BY 2
LIMIT 100


SELECT COUNT(*), COUNT(DISTINCT HADM_ID), SUBJECT_ID
FROM `data-science-269316.bd4h.hf_admissions_notes`
GROUP BY SUBJECT_ID
ORDER BY 2 DESC


SELECT *
FROM `data-science-269316.bd4h.hf_admissions_notes`
WHERE SUBJECT_ID = 11318



-- GET ANY READMISSIONS 
CREATE OR REPLACE TABLE `data-science-269316.bd4h.hf_admission_followed_by_readmission`
AS
SELECT HADM_ID, SUBJECT_ID, ADMITTIME,DISCHTIME,NextAdmitTime
FROM 
(
  SELECT HADM_ID, SUBJECT_ID, ADMITTIME,DISCHTIME
         LEAD(ADMITTIME) OVER (PARTITION BY SUBJECT_ID ORDER BY ADMITTIME) as NextAdmitTime
  FROM `data-science-269316.bd4h.hf_admissions`
) as Admissions
WHERE NextAdmitTime IS NOT NULL
ORDER BY SUBJECT_ID;

SELECT COUNT(*), COUNT(DISTINCT HADM_ID), COUNT(DISTINCT SUBJECT_ID)
FROM `data-science-269316.bd4h.hf_admission_followed_by_readmission`
--3,604 3,604 2,065




-- GET ANY READMISSIONS WITH NOTES
CREATE OR REPLACE TABLE `data-science-269316.bd4h.hf_readmission_notes`
AS
SELECT
  AdmissionsWithNext.HADM_ID,
  AdmissionsWithNext.SUBJECT_ID,
  AdmissionsWithNext.ADMITTIME,
  AdmissionsWithNext.DISCHTIME,
  AdmissionsWithNext.NextAdmitTime
FROM (
  SELECT HADM_ID, SUBJECT_ID,ADMITTIME,DISCHTIME,
         LEAD(ADMITTIME) OVER (PARTITION BY SUBJECT_ID ORDER BY ADMITTIME) as NextAdmitTime
  FROM `data-science-269316.bd4h.hf_admissions`
) as AdmissionsWithNext
INNER JOIN `data-science-269316.bd4h.hf_admissions_notes` notes ON AdmissionsWithNext.HADM_ID = notes.HADM_ID
WHERE AdmissionsWithNext.NextAdmitTime IS NOT NULL
ORDER BY AdmissionsWithNext.SUBJECT_ID, AdmissionsWithNext.ADMITTIME;

SELECT COUNT(*), COUNT(DISTINCT HADM_ID), COUNT(DISTINCT SUBJECT_ID)
FROM `data-science-269316.bd4h.hf_readmission_notes`
--3543 3543 2047

SELECT *
FROM `data-science-269316.bd4h.hf_readmission_notes`
WHERE SUBJECT_ID = 11318


-- GET 30DAY READMISSIONS 
CREATE OR REPLACE TABLE `data-science-269316.bd4h.hf_30day_readmission`
AS
SELECT
  HADM_ID,
  SUBJECT_ID,
  ADMITTIME,
  DISCHTIME,
  NextAdmitTime,
  TIMESTAMP_DIFF(NextAdmitTime, DISCHTIME, HOUR) AS HOURS_diff
FROM (
  SELECT
    HADM_ID,
    SUBJECT_ID,
    ADMITTIME,
    DISCHTIME,
    LEAD(ADMITTIME) OVER (PARTITION BY SUBJECT_ID ORDER BY ADMITTIME) as NextAdmitTime
  FROM `data-science-269316.bd4h.hf_admissions`
) as AdmissionsWithNext
WHERE NextAdmitTime IS NOT NULL
--AND DATE_DIFF(DATE(NextAdmitTime), DATE(DISCHTIME), DAY) <= 30
--AND DATEDIFF(DATE(NextAdmitTime), DATE(DISCHTIME))
AND TIMESTAMP_DIFF(NextAdmitTime, DISCHTIME, HOUR) <= (31 * 24)
ORDER BY SUBJECT_ID, ADMITTIME;


SELECT COUNT(*), COUNT(DISTINCT HADM_ID), COUNT(DISTINCT SUBJECT_ID)
FROM `data-science-269316.bd4h.hf_30day_readmission`
--969 969 737

-- GET 30DAY READMISSIONS WITH NOTES 
CREATE OR REPLACE TABLE `data-science-269316.bd4h.hf_30day_readmission_notes`
AS
SELECT
  AdmissionsWithNext.HADM_ID,
  AdmissionsWithNext.SUBJECT_ID,
  AdmissionsWithNext.ADMITTIME,
  AdmissionsWithNext.DISCHTIME,
  AdmissionsWithNext.NextAdmitTime
FROM (
  SELECT
    HADM_ID,
    SUBJECT_ID,
    ADMITTIME,
    DISCHTIME,
    LEAD(ADMITTIME) OVER (PARTITION BY SUBJECT_ID ORDER BY ADMITTIME) as NextAdmitTime
  FROM `data-science-269316.bd4h.hf_admissions`
) as AdmissionsWithNext
INNER JOIN `data-science-269316.bd4h.hf_admissions_notes` notes
ON AdmissionsWithNext.HADM_ID = notes.HADM_ID
WHERE AdmissionsWithNext.NextAdmitTime IS NOT NULL
AND TIMESTAMP_DIFF(AdmissionsWithNext.NextAdmitTime, AdmissionsWithNext.DISCHTIME, HOUR) < 24 * 31
ORDER BY AdmissionsWithNext.SUBJECT_ID, AdmissionsWithNext.ADMITTIME;


SELECT COUNT(*), COUNT(DISTINCT HADM_ID), COUNT(DISTINCT SUBJECT_ID)
FROM `data-science-269316.bd4h.hf_30day_readmission_notes`
--962 962 733



--#################### FINAL DATASET WITH TARGET LABELS ##########################

CREATE OR REPLACE TABLE `data-science-269316.bd4h.hf_admission_notes_with_target_labels`
AS
SELECT HADM_ID, SUBJECT_ID,TEXT,DESCRIPTION ,ADMITTIME,
       CASE WHEN HADM_ID IN (SELECT HADM_ID FROM `data-science-269316.bd4h.hf_readmission_notes`) THEN 1 ELSE 0 END AS READMISSION,
       CASE WHEN HADM_ID IN (SELECT HADM_ID FROM `data-science-269316.bd4h.hf_30day_readmission_notes`) THEN 1 ELSE 0 END AS READMISSION_30DAYS,
FROM `data-science-269316.bd4h.hf_admissions_notes`


SELECT COUNT(*), COUNT(DISTINCT HADM_ID), COUNT(DISTINCT SUBJECT_ID)
FROM `data-science-269316.bd4h.hf_admission_notes_with_target_labels`
--13,746 13,746 10,244


SELECT COUNT(*), COUNT(DISTINCT HADM_ID), SUM(READMISSION),SUM(READMISSION_30DAYS)
FROM `data-science-269316.bd4h.hf_admission_notes_with_target_labels`
--13,746  13,746 3,543 962

SELECT *
FROM `data-science-269316.bd4h.hf_admission_notes_with_target_labels`
WHERE READMISSION = 1

