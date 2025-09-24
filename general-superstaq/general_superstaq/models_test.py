from __future__ import annotations

import pydantic
import pytest

import general_superstaq as gss

"""This test file will be updated once the new server (v0.3.0) client is implemented.
For now, a single test is sufficient to check that `pydantic` is loading models and working as
expected.
"""


def test_user_token_response() -> None:
    gss.models.UserTokenResponse(
        email="valid.email@infleqtion.com",
        token="abc",
    )
    with pytest.raises(pydantic.ValidationError):
        gss.models.UserTokenResponse(
            email="this is not an email",
            token="abc",
        )


def test_external_provider_credentials() -> None:
    credentials = gss.models.ExternalProviderCredentials()
    assert credentials.cq_token is None
    assert credentials.cq_project_id is None
    assert credentials.cq_org_id is None

    # New-style CQ credentials
    options_new = {"cq_token": "token", "cq_project_id": "123", "cq_org_id": "456"}
    credentials = gss.models.ExternalProviderCredentials(**options_new)
    assert credentials.cq_token == "token"
    assert credentials.cq_project_id == "123"
    assert credentials.cq_org_id == "456"

    # Old-style CQ credentials
    options_old = {"cq_token": {"access_token": "token"}, "project_id": "123", "org_id": "456"}
    credentials = gss.models.ExternalProviderCredentials(**options_old)
    assert credentials.cq_token == "token"
    assert credentials.cq_project_id == "123"
    assert credentials.cq_org_id == "456"
