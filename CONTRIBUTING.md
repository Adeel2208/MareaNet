# Contributing to MAREA-Net

Thank you for your interest in contributing to MAREA-Net! This document provides guidelines for contributing to the project.

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/yourusername/marea-net.git`
3. Create a branch: `git checkout -b feature/your-feature-name`
4. Make your changes
5. Test your changes
6. Commit: `git commit -m "Add your feature"`
7. Push: `git push origin feature/your-feature-name`
8. Create a Pull Request

## Development Setup

```bash
# Clone repository
git clone https://github.com/yourusername/marea-net.git
cd marea-net

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e .
pip install -r requirements.txt

# Install development dependencies
pip install pytest black flake8 mypy
```

## Code Style

We follow PEP 8 style guidelines with some modifications:

- Line length: 100 characters
- Use 4 spaces for indentation
- Use type hints where appropriate

### Formatting

Use `black` for code formatting:

```bash
black marea_net/ train.py inference.py evaluate.py
```

### Linting

Use `flake8` for linting:

```bash
flake8 marea_net/ --max-line-length=100
```

## Testing

### Running Tests

```bash
pytest tests/
```

### Writing Tests

Place tests in the `tests/` directory:

```python
# tests/test_model.py
import tensorflow as tf
from marea_net import build_marea_net, Config

def test_model_build():
    config = Config()
    model = build_marea_net(config)
    assert model is not None
    assert len(model.outputs) == 3

def test_model_inference():
    config = Config()
    model = build_marea_net(config)
    x = tf.random.normal([1, 320, 320, 3])
    seg, aux, plant = model(x, training=False)
    assert seg.shape == (1, 320, 320, 8)
```

## Documentation

### Docstrings

Use Google-style docstrings:

```python
def my_function(param1, param2):
    """
    Brief description of function.
    
    Longer description if needed.
    
    Args:
        param1 (type): Description of param1
        param2 (type): Description of param2
        
    Returns:
        type: Description of return value
        
    Raises:
        ValueError: When something goes wrong
    """
    pass
```

### Updating Documentation

- Update relevant `.md` files in `docs/`
- Update `README.md` if adding new features
- Update `API.md` for new public functions

## Pull Request Guidelines

### Before Submitting

- [ ] Code follows style guidelines
- [ ] Tests pass
- [ ] New tests added for new features
- [ ] Documentation updated
- [ ] Commit messages are clear and descriptive

### PR Description

Include:

1. **What**: Brief description of changes
2. **Why**: Motivation for changes
3. **How**: Technical details if complex
4. **Testing**: How you tested the changes
5. **Screenshots**: If UI/visualization changes

### Example PR Template

```markdown
## Description
Brief description of what this PR does.

## Motivation
Why is this change needed?

## Changes
- Change 1
- Change 2
- Change 3

## Testing
How did you test these changes?

## Checklist
- [ ] Tests pass
- [ ] Documentation updated
- [ ] Code formatted with black
- [ ] No linting errors
```

## Types of Contributions

### Bug Fixes

- Include test that reproduces the bug
- Explain the fix in PR description
- Reference issue number if applicable

### New Features

- Discuss in an issue first
- Include tests
- Update documentation
- Add example usage

### Documentation

- Fix typos, improve clarity
- Add examples
- Update API documentation

### Performance Improvements

- Include benchmarks showing improvement
- Ensure no accuracy regression
- Document trade-offs

## Reporting Issues

### Bug Reports

Include:

1. **Description**: Clear description of the bug
2. **Steps to Reproduce**: Minimal code to reproduce
3. **Expected Behavior**: What should happen
4. **Actual Behavior**: What actually happens
5. **Environment**: OS, Python version, TensorFlow version, GPU info
6. **Logs**: Relevant error messages or logs

### Feature Requests

Include:

1. **Description**: What feature you'd like
2. **Motivation**: Why it's useful
3. **Proposed Solution**: How it might work
4. **Alternatives**: Other approaches considered

## Code Review Process

1. Maintainer reviews PR
2. Feedback provided via comments
3. Author addresses feedback
4. Maintainer approves and merges

## Community Guidelines

- Be respectful and constructive
- Welcome newcomers
- Focus on the code, not the person
- Assume good intentions
- Ask questions if unclear

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

## Questions?

- Open an issue for questions
- Tag with `question` label
- Check existing issues first

## Recognition

Contributors will be acknowledged in:
- `README.md` contributors section
- Release notes for significant contributions

Thank you for contributing to MAREA-Net! 🌊
