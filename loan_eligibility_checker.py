class LoanApplicant:
    def __init__(self, age, income, employment_type, loan_type, credit_score, work_experience=None):
        self.age = age
        self.income = income
        self.employment_type = employment_type  # 'salaried' or 'self-employed'
        self.loan_type = loan_type  # 'personal' or 'home'
        self.credit_score = credit_score
        self.work_experience = work_experience  # in years, applicable for self-employed

class LoanEligibilityChecker:
    def __init__(self, applicant):
        self.applicant = applicant

    def check_personal_loan_eligibility(self):
        # Age Criteria
        if not (22 <= self.applicant.age <= 58):
            return False, "Age must be between 22 and 58 years for personal loan eligibility."

        # Income and Employment Criteria
        if self.applicant.employment_type == 'salaried':
            if self.applicant.income < 22000:
                return False, "Minimum monthly income for salaried individuals is ₹22,000."
        elif self.applicant.employment_type == 'self-employed':
            if self.applicant.income < 25000:
                return False, "Minimum monthly income for self-employed individuals is ₹25,000."
            if self.applicant.work_experience is None or self.applicant.work_experience < 2:
                return False, "Self-employed individuals must have at least 2 years of business continuity."
        else:
            return False, "Invalid employment type. Must be 'salaried' or 'self-employed'."

        # Credit Score Criteria
        if self.applicant.credit_score < 650:
            return False, "A minimum credit score of 650 is required for personal loans."

        return True, "Eligible for personal loan."

    def check_home_loan_eligibility(self):
        # Age Criteria
        if self.applicant.employment_type == 'salaried':
            if not (23 <= self.applicant.age <= 63):
                return False, "Age must be between 23 and 63 years for salaried individuals applying for a home loan."
        elif self.applicant.employment_type == 'self-employed':
            if not (23 <= self.applicant.age <= 70):
                return False, "Age must be between 23 and 70 years for self-employed individuals applying for a home loan."
        else:
            return False, "Invalid employment type. Must be 'salaried' or 'self-employed'."

        # Income Criteria
        if self.applicant.income < 50000:
            return False, "Minimum monthly income for home loan applicants is ₹50,000."

        # Credit Score Criteria
        if self.applicant.credit_score < 750:
            return False, "A minimum credit score of 750 is required for home loans."

        return True, "Eligible for home loan."

    def evaluate(self):
        if self.applicant.loan_type == 'personal':
            return self.check_personal_loan_eligibility()
        elif self.applicant.loan_type == 'home':
            return self.check_home_loan_eligibility()
        else:
            return False, "Invalid loan type. Must be 'personal' or 'home'."

# Example usage:
applicant = LoanApplicant(age=30, income=60000, employment_type='salaried', loan_type='personal', credit_score=720)
checker = LoanEligibilityChecker(applicant)
is_eligible, message = checker.evaluate()
print(message)
