# Validate Dockerfile syntax
with open('Dockerfile', 'r') as f:
    content = f.read()

checks = [
    ('FROM python:3.10-slim' in content, 'FROM instruction'),
    ('WORKDIR /app' in content, 'WORKDIR instruction'),
    ('COPY requirements.txt' in content, 'COPY instruction'),
    ('RUN pip install' in content, 'RUN pip install'),
    ('EXPOSE 7860' in content, 'EXPOSE port'),
    ('CMD [' in content, 'CMD instruction'),
    ('HEALTHCHECK' in content, 'HEALTHCHECK instruction'),
]

print('Dockerfile validation:')
all_pass = True
for check, name in checks:
    status = 'PASS' if check else 'FAIL'
    if not check:
        all_pass = False
    print(f'  {status}: {name}')

print()
if all_pass:
    print('All checks passed - Dockerfile looks valid')
else:
    print('Some checks failed - please review the Dockerfile')
