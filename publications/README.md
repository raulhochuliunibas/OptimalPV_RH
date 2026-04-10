# Publications & Documentation

This directory contains working papers, presentations, and outputs related to the OptimalPV_RH project.

## Structure

- **`working_papers/`**: Working papers, manuscript versions, and documentation organized by code version
- **`presentations/`**: Slides, conference presentations, and workshop materials organized by event/date
- **`htmls/`**: Generated HTML outputs from analyses and visualizations (ignored from git)

## Organization Convention

For each code version or publication, create a subdirectory following the pattern:
```
working_papers/v{version}_{description}/
presentations/{year}_{event}_{description}/
```

Example:
```
working_papers/v1.0_PV_allocation_methodology/
presentations/2024_ECP_Conference_Solar_Integration/
```

## File Types

- **.csv**: Included in repo for reproducibility and tracking changes
- **.md**: Markdown documentation files (included in repo)
- **.html**: Generated outputs (ignored, stored locally in `htmls/`)

## Version Control

- CSV results and documentation are tracked in git
- HTML outputs are excluded to keep the repository lean
- Large binary files (PDFs, graphics) should be considered for a separate release system
