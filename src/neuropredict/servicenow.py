from __future__ import annotations

import os
import uuid

import requests


class ServiceNowClient:
    """ServiceNow Table API client with timeout recovery."""

    def __init__(self):
        self.instance = os.environ["SERVICENOW_INSTANCE"].rstrip("/")
        self.user = os.environ["SERVICENOW_USER"]
        self.password = os.environ["SERVICENOW_PASSWORD"]
        self.table = os.getenv("SERVICENOW_TABLE", "incident")

    def _auth(self):
        return (self.user, self.password)

    def create_incident(
        self,
        short_description: str,
        description: str,
        severity: str = "3",
    ) -> dict:

        correlation_id = f"neuropredict-{uuid.uuid4().hex}"

        url = f"{self.instance}/api/now/table/{self.table}"

        payload = {
            "short_description": short_description,
            "description": description,
            "severity": severity,
            "correlation_id": correlation_id,
        }

        try:
            response = requests.post(
                url,
                auth=self._auth(),
                headers={
                    "Accept": "application/json",
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=(10, 20),
            )

            response.raise_for_status()
            return response.json()

        except requests.exceptions.Timeout:
            # ServiceNow may have created the record but failed
            # to return the response before the client timeout.
            lookup_url = (
                f"{self.instance}"
                f"/api/now/table/{self.table}"
            )

            params = {
                "sysparm_query": (
                    f"correlation_id={correlation_id}"
                ),
                "sysparm_fields": (
                    "number,sys_id,short_description"
                ),
                "sysparm_limit": "1",
            }

            lookup = requests.get(
                lookup_url,
                params=params,
                auth=self._auth(),
                headers={
                    "Accept": "application/json",
                },
                timeout=(10, 20),
            )

            lookup.raise_for_status()

            result = lookup.json().get("result", [])

            if result:
                return {
                    "result": result[0],
                    "recovered_after_timeout": True,
                }

            raise