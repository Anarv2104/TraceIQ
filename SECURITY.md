# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 0.2.x   | :white_check_mark: |
| < 0.2   | :x:                |

## Reporting a Vulnerability

If you discover a security vulnerability in TraceIQ, please report it by:

1. **Email**: Open an issue on GitHub with the label "security" (do not include sensitive details in public issues)
2. **GitHub Security Advisories**: Use GitHub's private vulnerability reporting feature

### What to Include

- Description of the vulnerability
- Steps to reproduce
- Potential impact
- Suggested fix (if any)

### Response Timeline

- **Initial response**: Within 48 hours
- **Status update**: Within 1 week
- **Resolution**: Depends on severity

### Severity Levels

- **Critical**: Remote code execution, data exfiltration
- **High**: Privilege escalation, data corruption
- **Medium**: Information disclosure, denial of service
- **Low**: Minor issues with limited impact

## Security Considerations

### Data Storage

- SQLite databases may contain sensitive agent interaction data
- Use appropriate file permissions on database files
- Consider encrypting database files at rest for sensitive deployments

### Embeddings

- When using sentence-transformers, model weights are downloaded from Hugging Face
- Verify model integrity if operating in high-security environments
- Consider using local models for air-gapped deployments

### Dependencies

- Keep dependencies updated to receive security patches
- Use `pip-audit` or similar tools to scan for known vulnerabilities
