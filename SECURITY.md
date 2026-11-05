# Security Policy

## Supported versions

Only the latest code on the `main` branch receives security fixes.

| Version        | Supported |
| -------------- | --------- |
| `main` (latest) | ✅        |
| Older commits / releases | ❌ |

TerraTexture requires Python 3.11 or newer. Older Python versions are not supported, because several dependencies (for example Pillow 13+) have dropped them.

## Reporting a vulnerability

**Please do not open a public issue for security problems.**

Report vulnerabilities privately through GitHub:

1. Go to the **Security** tab of this repository.
2. Click **Report a vulnerability**.
3. Include:
   - a description of the issue and its impact
   - steps to reproduce, or a proof of concept
   - the affected commit or version
   - any suggested fix, if you have one

### What to expect

- **Acknowledgement:** within 7 days of your report.
- **Assessment:** an initial assessment and severity rating within 14 days.
- **Fix:** confirmed issues are patched on `main` as quickly as possible. You'll be kept updated on progress.
- **Credit:** you'll be credited in the fix commit or advisory unless you'd rather stay anonymous.

If a report is declined, you'll get an explanation of why.

## Dependency security

Dependencies are monitored with Dependabot security alerts and updates. Vulnerable dependencies are upgraded to the earliest fixed version that's compatible with the supported Python version.
