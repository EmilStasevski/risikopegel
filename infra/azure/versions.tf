terraform {
  required_version = ">= 1.9"
  required_providers {
    azurerm = {
      source  = "hashicorp/azurerm"
      version = "= 5.13.0" # released 2026-11-05 (>= 2 weeks before adoption); see docs/versions.md
    }
  }
}

provider "azurerm" {
  features {}
  subscription_id = var.subscription_id
  # azurerm 5.0 no longer registers resource providers by default: register exactly the two we need.
  resource_provider_registrations = "none"
  resource_providers_to_register  = ["Microsoft.App", "Microsoft.OperationalInsights"]
}
