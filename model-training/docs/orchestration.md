# Rhythm AI – MLOps Job Orchestration & Scheduling

## What

- Automate scheduled or event-driven jobs (training, evaluation, deployment, compliance export, etc)
- Each job is a shell script or command file in `jobs/`
- Scheduler discovers and runs jobs, logging all actions for audit

## How

- Place a `.job` file (bash script or command) in `jobs/`
- Scheduler runs pending jobs and marks them as `.done`
- All job events are audit logged

## Example

```bash
# jobs/retrain_model.job
python train.py --config model.yaml
```

```bash
python orchestration/job_scheduler.py
```

- Integrate with CI/CD or K8s CronJobs for large-scale workflows

---

*Simple, auditable orchestration—ready for enterprise pipelines.*