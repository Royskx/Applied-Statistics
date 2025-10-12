# Applied Statistics Course

> 📚 **Course Materials for Students**

Welcome to Applied Statistics! This repository contains all the materials you need for this course.

## 🎯 Course Overview

An introduction to applied statistics with materials released progressively throughout the semester.

**Currently available:**
- **Statistical modeling** and exploratory data analysis
- **Estimation methods** (Maximum Likelihood, Method of Moments)
- **Estimator properties** (bias, variance, confidence intervals, bootstrap)

**Coming soon:**
- Hypothesis testing and applications
- Final project guidelines

### Prerequisites
- Basic probability theory (random variables, distributions)
- Python programming fundamentals
- Linear algebra basics

## 📁 What's in This Repository

### 📖 Lessons (`lessons/`)

Each lesson folder contains:
- **material.md** or **PDF** - Lesson content and theory
- **exercises/** - Practice problems (when applicable)
- **data/** - Datasets for the lesson

| Lesson | Topic | Materials |
|--------|-------|-----------|
| 00 | Welcome & Introduction | Introduction to the course |
| 01 | Statistical Modeling | Random variables, distributions, EDA |
| 02 | Statistical Learning | MLE, Method of Moments, Fisher information |
| 03 | Estimator Properties | Bias, variance, MSE, confidence intervals, bootstrap |

> **Note**: Additional lessons (04-06) will be released throughout the semester.

### 🧪 Lab Sessions (`labs/Subjects/`)

Interactive Jupyter notebooks for hands-on practice:

1. **Random Variables** - Working with distributions and simulation
2. **Maximum Likelihood Estimation** - Parameter estimation

> **Note**: Additional lab sessions will be released as we progress through the course.

### 📊 Data (`shared/data/`)

Sample datasets for exercises and projects:
- `heights_weights_sample.csv` - Anthropometric data
- `ab_test_clicks.csv` - A/B testing data
- `manufacturing_defects.csv` - Quality control data
- Additional datasets in lesson-specific folders

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/stephane-rivaud/Applied-Statistics.git
cd Applied-Statistics
```

### 2. Set Up Python Environment

**Create a virtual environment** (recommended):
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

**Install required packages**:
```bash
pip install -r requirements.txt
```

This installs:
- NumPy - Numerical computing
- Pandas - Data manipulation
- Matplotlib & Seaborn - Visualization
- SciPy - Statistical functions
- Scikit-learn - Machine learning
- Jupyter - Interactive notebooks

### 3. Launch Jupyter

```bash
jupyter lab
```

Navigate to `labs/Subjects/` and open a notebook to get started!

### 4. Alternative: Google Colab

You can also run notebooks in Google Colab:
1. Upload the notebook to Google Drive
2. Open with Google Colaboratory
3. Install packages: `!pip install numpy pandas matplotlib scipy scikit-learn`

## 📚 How to Use This Repository

### Weekly Workflow

1. **Before class**: Read the lesson material in `lessons/XX-topic/`
2. **During class**: Follow along with slides and examples
3. **After class**: Complete the lab assignment in `labs/Subjects/`
4. **Practice**: Work through exercises in `lessons/XX-topic/exercises/`

### Completing Lab Assignments

1. Open the notebook in `labs/Subjects/`
2. Read instructions carefully
3. Write your code in the designated cells
4. Test your code thoroughly
5. Save your work regularly
6. Submit according to instructor guidelines

### Working with Data

```python
# Example: Loading a dataset
import pandas as pd

# From shared data folder
df = pd.read_csv('shared/data/heights_weights_sample.csv')

# From lesson-specific folder (when available)
# df = pd.read_csv('lessons/03-estimator-properties/data/dataset.csv')
```

## 🔄 Getting Updates

New materials are released throughout the semester. To get updates:

```bash
# Make sure you've committed or saved your work first!
git pull origin public-main
```

**Important**: If you've modified any files, save your changes first or they may be overwritten.

## 📖 Course Schedule & Syllabus

See **[syllabus.md](syllabus.md)** for:
- Detailed lesson schedule
- Learning outcomes
- Grading criteria
- Course policies
- Important dates

### Assessment

- **Lab Assignments**: Hands-on exercises throughout the course
- **Final Project**: Data analysis project with report and presentation

> See [syllabus.md](syllabus.md) for detailed grading breakdown and deadlines.

## 📚 Required Textbooks

### Primary References

1. **All of Statistics** by Larry Wasserman
   - Comprehensive coverage of statistical theory
   - [PDF Link](https://www.stat.cmu.edu/~brian/valerie/617-2022/0%20-%20books/2004%20-%20wasserman%20-%20all%20of%20statistics.pdf)

2. **A Modern Introduction to Probability and Statistics** by Dekking et al.
   - Gentle introduction with applications
   - [PDF Link](https://cis.temple.edu/~latecki/Courses/CIS2033-Spring13/Modern_intro_probability_statistics_Dekking05.pdf)

### Additional Resources

- **Think Stats** by Allen Downey - Python-based statistics
- **Statistical Learning** MOOC - Stanford University
- **OpenIntro Statistics** - Free online textbook

## 💻 Software & Tools

### Required

- **Python 3.9 or later**
- **Jupyter Notebook/Lab** (included in requirements.txt)

### Python Libraries (all in requirements.txt)

| Library | Purpose |
|---------|---------|
| NumPy | Numerical arrays and operations |
| Pandas | Data manipulation and analysis |
| Matplotlib | Basic plotting |
| Seaborn | Statistical visualizations |
| SciPy | Statistical functions and tests |
| Scikit-learn | Machine learning tools |

### Optional Tools

- **Git** - Version control (for getting updates)
- **VS Code** - Code editor with Jupyter support
- **Google Colab** - Cloud-based Jupyter environment

## ❓ Getting Help

### During Office Hours

**Instructor Office Hours**: [TBD - Check course website]
- Come prepared with specific questions
- Bring your code and error messages
- Review material before attending

### Online Resources

- **Course Forum**: [TBD - Add link if available]
- **Email**: [Instructor email]
- **Stack Overflow** - For Python/technical questions (tag: statistics, python, pandas)

### Tips for Getting Help

1. **Try to solve it first** - Review lesson materials and documentation
2. **Search for similar problems** - Check forum, Stack Overflow
3. **Create a minimal example** - Isolate the specific issue
4. **Include error messages** - Copy the full error traceback
5. **Explain what you've tried** - Show your attempted solutions

## 💡 Tips for Success

### General Advice

- ✅ **Start assignments early** - Don't wait until the deadline
- ✅ **Read error messages carefully** - They often tell you exactly what's wrong
- ✅ **Comment your code** - Explain your reasoning
- ✅ **Test incrementally** - Don't write everything before testing
- ✅ **Use version control** - Save your work frequently
- ✅ **Attend lectures** - Concepts build on each other

### Python Tips

```python
# Good practices for data analysis

# 1. Always inspect your data first
df.head()
df.info()
df.describe()

# 2. Handle missing values
df.isnull().sum()  # Check for missing values
df = df.dropna()   # Remove or...
df = df.fillna(df.mean())  # Impute

# 3. Visualize before analyzing
import matplotlib.pyplot as plt
df['column'].hist()
plt.show()

# 4. Document your assumptions
# Assume normal distribution based on histogram
# Sample size n=100 is sufficient for CLT

# 5. Validate your results
# Double-check calculations
# Compare with analytical solutions when possible
```

### Statistical Analysis Tips

- Always start with exploratory data analysis (EDA)
- Check assumptions before applying methods
- Visualize your results
- Interpret p-values carefully (they're not everything!)
- Report confidence intervals, not just point estimates
- Consider practical significance, not just statistical significance

## 📋 Academic Integrity

### Collaboration Policy

- ✅ **Allowed**: Discussing concepts and approaches with classmates
- ✅ **Allowed**: Getting help during office hours
- ✅ **Allowed**: Using online resources for understanding concepts
- ❌ **Not allowed**: Copying code from others
- ❌ **Not allowed**: Sharing your solution code before deadlines
- ❌ **Not allowed**: Using code without understanding or attribution

### What to Submit

- **Your own work**: Write your own code and explanations
- **Properly cited**: Credit any sources, discussions, or help received
- **Well-documented**: Comment your code and explain your reasoning

### Consequences

Violations of academic integrity will be handled according to university policies and may result in:
- Zero on the assignment
- Failing the course
- Disciplinary action

When in doubt, ask the instructor!

## 📅 Important Dates

*Check syllabus.md and course website for current dates*

| Date | Event |
|------|-------|
| Week 1 | Course introduction, Lesson 01 |
| Week 2 | Lesson 02, Lab 1 due |
| Week 3 | Lesson 03, Lab 2 due |
| TBD | Additional lessons and assignments |

> **Note**: Full schedule will be updated as the course progresses. Check the repository regularly for updates.

## 🔗 Useful Links

- **Course Website**: [TBD]
- **Assignment Submission**: [TBD]
- **Discussion Forum**: [TBD]
- **Gradebook**: [TBD]

## 📞 Contact Information

- **Instructor**: [Name]
- **Email**: [Email]
- **Office Hours**: [Schedule and location]
- **Course Website**: [URL]

## 🆘 Troubleshooting

### Common Issues

**Problem**: `ModuleNotFoundError: No module named 'pandas'`
```bash
# Solution: Install requirements
pip install -r requirements.txt
```

**Problem**: Jupyter kernel keeps dying
```bash
# Solution: Increase memory or simplify computation
# Or try running on Google Colab
```

**Problem**: Git pull conflicts
```bash
# Solution: Save your work elsewhere, then reset
git stash  # Save your changes
git pull   # Get updates
git stash pop  # Restore your changes
```

**Problem**: Can't run cells in Jupyter
```bash
# Solution: Make sure kernel is running
# Click "Kernel" → "Restart Kernel"
```

---

**Repository**: https://github.com/stephane-rivaud/Applied-Statistics
**Last Updated**: October 11, 2025
**Good luck and enjoy the course! 📊🎓**
