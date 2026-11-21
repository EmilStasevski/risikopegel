# Azure deployment

A small read-only API on **Azure Container Apps**, provisioned with **Terraform** (`infra/azure/`). The image is built by GitHub Actions and pushed to the GitHub Container Registry. User decision on 2026-11-21: Azure in scope (PLAN §7 had it as optional).

## What runs where

- **Local or CI:** `Dockerfile` builds a ~300 MB image with FastAPI and uvicorn only.
  - In the image: the generated `reports/` and the aggregate `results/` JSON.
  - Not in the image: `data/` (licensed rows, which several licences forbid redistributing) and trained models. `.dockerignore` keeps these out of the build context.
- **GitHub Actions** (`.github/workflows/ci.yml`):
  - `terraform fmt -check` and `validate`, with no Azure access;
  - an image build plus a container smoke test (`/health`, one report);
  - a push to `ghcr.io/<owner>/risikopegel-api:<sha>` from `main` only.
- **Azure**, all in one resource group:
  - Log Analytics workspace: 30-day retention, 0.1 GB/day ingestion cap;
  - Container Apps environment (consumption);
  - one Container App: 0.25 vCPU / 0.5 GiB, **min replicas 0** (scale to zero), max 1, HTTP probes on `/health`;
  - a monthly resource-group budget with e-mail alerts at 50/80/100 %.

## Endpoints

`/health`, `/` (index), `/reports/{name}.md`, `/results/{path}.json`. Module E adds the risk-bearing-capacity (RTF) calculator.

## Cost

Verified 2026-11-21, [Microsoft Learn: Container Apps billing](https://learn.microsoft.com/azure/container-apps/billing):
- The consumption plan's free monthly grant per subscription is 180,000 vCPU-seconds, 360,000 GiB-seconds and 2 million requests. Health-probe requests are not billed.
- A scale-to-zero demo app with occasional traffic should stay inside that grant.
- Log Analytics ingestion is capped by the daily quota.
- The budget alert is the safety net. Per-second prices vary by region and are not taken from memory.

## Deploying

Nothing is created in Azure without an explicit go (PLAN §0 rule 3).

```bash
az login                                  # interactive
cd infra/azure
cp terraform.tfvars.example terraform.tfvars   # subscription_id, image, budget_start
export TF_VAR_alert_email=...             # not committed
terraform init && terraform plan -out=tfplan   # review
terraform apply tfplan
terraform output api_url
terraform destroy                          # removes everything again
```

State is local (`terraform.tfstate`, git-ignored) for now. A remote backend in an Azure Storage account and deployment from CI via OIDC (workload identity federation, no stored secrets) are the next steps. Both create Entra ID and Azure objects, so they need a separate go.

## Versions

Terraform 1.17.2; azurerm provider 5.13.0. That release is at least two weeks old; 5.14.0, from 2026-11-13, was too new. azurerm 5.0 no longer registers resource providers by default, so `versions.tf` registers `Microsoft.App` and `Microsoft.OperationalInsights` explicitly.
