variable "subscription_id" {
  description = "Azure subscription to deploy into"
  type        = string
}

variable "location" {
  description = "Azure region"
  type        = string
  default     = "germanywestcentral"
}

variable "name" {
  description = "Name prefix for all resources"
  type        = string
  default     = "risikopegel"
}

variable "image" {
  description = "Public container image (GitHub Container Registry), e.g. ghcr.io/<owner>/risikopegel-api:<sha>"
  type        = string
}

variable "budget_eur" {
  description = "Monthly budget on the resource group; alert e-mails at 50/80/100 %"
  type        = number
  default     = 5
}

variable "alert_email" {
  description = "Recipient of budget alerts (kept out of the repo: pass via TF_VAR_alert_email)"
  type        = string
  sensitive   = true
}

variable "budget_start" {
  description = "First day of the budget period, YYYY-MM-01T00:00:00Z"
  type        = string
}
