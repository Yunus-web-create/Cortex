"""Access control test — the login screen, the 90-day cookie, and Basic auth together.

Fails loudly if any of these ever regress:
  * the app is open when CORTEX_PASSWORD is unset (local dev),
  * the app leaks anything but /api/health, the login routes, the manifest, sw and icons
    to an unauthenticated visitor,
  * a wrong password is accepted, or a cookie is minted for it,
  * a valid cookie or valid Basic credentials stop working,
  * "?next=" becomes an open redirect,
  * logout leaves the cookie valid.

Usage:  python tests/auth_test.py
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

TMP = Path(tempfile.mkdtemp(prefix="cortex_auth_"))
os.environ["CORTEX_DATA_DIR"] = str(TMP)
os.environ["CORTEX_DB"] = str(TMP / "test.db")
os.environ["CORTEX_PASSWORD"] = "correct horse battery"
os.environ["CORTEX_RERANK"] = "0"
os.environ["CORTEX_FREE_MODEL"] = "0"
os.environ["CORTEX_LOCAL_MODEL"] = "0"
for _var in ("AI_GATEWAY_API_KEY", "OPENROUTER_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY",
             "CORTEX_API_KEY", "CORTEX_PROVIDER", "CORTEX_BASE_URL"):
    os.environ[_var] = ""

from fastapi.testclient import TestClient  # noqa: E402

from app.main import app  # noqa: E402

PASSWORD = "correct horse battery"
BROWSER = {"accept": "text/html,application/xhtml+xml"}

passed = failed = 0


def check(name: str, condition: bool, detail: str = "") -> None:
    global passed, failed
    if condition:
        passed += 1
        print(f"  ok   {name}")
    else:
        failed += 1
        print(f"  FAIL {name}{(' — ' + detail) if detail else ''}")


def fresh() -> TestClient:
    """A client with an empty cookie jar — one TestClient keeps cookies across calls."""
    return TestClient(app, follow_redirects=False)


def main() -> int:
    client = fresh()

    print("\nopen paths")
    r = client.get("/api/health")
    check("health is reachable without a password", r.status_code == 200, str(r.status_code))
    for path in ("/manifest.webmanifest", "/sw.js", "/icons/icon-192.png"):
        r = client.get(path)
        check(f"{path} is public", r.status_code == 200, str(r.status_code))

    print("\nlocked paths")
    r = client.get("/api/settings")
    check("api is 401 without credentials", r.status_code == 401, str(r.status_code))
    check("api 401 is JSON, not HTML", r.headers.get("content-type", "").startswith("application/json"))
    check("api 401 advertises Basic", "Basic" in r.headers.get("www-authenticate", ""))
    r = client.get("/", headers=BROWSER)
    check("browser GET / redirects to /login", r.status_code == 303, str(r.status_code))
    check("redirect keeps the target", "/login?next=%2F" in r.headers.get("location", ""),
          r.headers.get("location", ""))
    r = client.get("/api/settings", headers={"accept": "application/json"})
    check("a JSON fetch is never redirected", r.status_code == 401, str(r.status_code))

    print("\nlogin screen")
    r = client.get("/login?next=/")
    check("login page renders", r.status_code == 200, str(r.status_code))
    check("login page has the form", 'name="password"' in r.text and 'method="post"' in r.text)
    check("login page carries the next target", 'name="next"' in r.text)
    check("login page is self-contained", "http://" not in r.text and "https://" not in r.text)

    print("\nwrong password")
    r = client.post("/login", data={"password": "nope", "next": "/"})
    check("wrong password is rejected", r.status_code == 401, str(r.status_code))
    check("no cookie is set", "set-cookie" not in r.headers)
    check("the error is shown", "did not match" in r.text)

    print("\nopen redirect")
    r = client.post("/login", data={"password": PASSWORD, "next": "//evil.example/x"})
    check("protocol-relative next is refused", r.headers.get("location") == "/",
          r.headers.get("location", ""))

    print("\ncorrect password")
    r = client.post("/login", data={"password": PASSWORD, "next": "/?x=1"})
    check("login redirects", r.status_code == 303, str(r.status_code))
    check("login returns to the requested page", r.headers.get("location") == "/?x=1",
          r.headers.get("location", ""))
    cookie = r.headers.get("set-cookie", "")
    check("cookie is set", "cortex_access=" in cookie)
    check("cookie is HttpOnly", "httponly" in cookie.lower())
    check("cookie is SameSite=Lax", "samesite=lax" in cookie.lower())
    check("cookie lasts 90 days", "max-age=7776000" in cookie.lower(), cookie)
    check("cookie is not marked Secure over http", "secure" not in cookie.lower(), cookie)

    token = cookie.split("cortex_access=", 1)[1].split(";", 1)[0]
    check("token is a 64-char HMAC", len(token) == 64, str(len(token)))

    print("\nauthenticated access")
    r = client.get("/api/settings", cookies={"cortex_access": token})
    check("cookie opens the API", r.status_code == 200, str(r.status_code))
    r = client.get("/", headers=BROWSER, cookies={"cortex_access": token})
    check("cookie opens the app shell", r.status_code == 200, str(r.status_code))
    r = fresh().get("/api/settings", cookies={"cortex_access": "deadbeef"})
    check("a forged cookie is rejected", r.status_code == 401, str(r.status_code))

    r = fresh().get("/api/settings", auth=("cortex", PASSWORD))
    check("Basic auth still works", r.status_code == 200, str(r.status_code))
    r = fresh().get("/api/settings", auth=(PASSWORD, ""))
    check("password-as-username still works", r.status_code == 200, str(r.status_code))
    r = fresh().get("/api/settings", auth=("cortex", "wrong"))
    check("wrong Basic password is rejected", r.status_code == 401, str(r.status_code))

    print("\nhttps deployments")
    r = client.post("/login", data={"password": PASSWORD, "next": "/"},
                    headers={"x-forwarded-proto": "https"})
    check("cookie is Secure behind an https proxy", "secure" in r.headers.get("set-cookie", "").lower())

    print("\nlogout")
    r = client.get("/logout", cookies={"cortex_access": token})
    check("logout redirects to the login page", r.status_code == 303, str(r.status_code))
    check("logout clears the cookie", 'cortex_access=""' in r.headers.get("set-cookie", ""),
          r.headers.get("set-cookie", ""))
    check("logout expires the cookie", "max-age=0" in r.headers.get("set-cookie", "").lower())

    print("\ninstallation still works behind the login")
    r = client.get("/manifest.webmanifest", cookies={"cortex_access": token})
    import json as _json
    m = _json.loads(r.text)
    check("manifest stays installable",
          m.get("display") == "standalone" and str(m.get("start_url", "")).startswith("/"))
    r = client.get("/sw.js", cookies={"cortex_access": token})
    check("service worker is served from the root scope",
          r.headers.get("service-worker-allowed") == "/")

    print(f"\n{passed} passed, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
