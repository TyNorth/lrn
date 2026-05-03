"""
Grammar Workbook Generator - Creates practice sentences and assessments
for each grade level based on grammar lesson modules.

Usage:
    from lrn.grammar_workbook import GrammarWorkbook
    workbook = GrammarWorkbook(grade=2)
    sentences = workbook.generate_practice("subject-verb agreement")
    assessment = workbook.generate_assessment("past tense")
"""

import sys
sys.path.insert(0, '/Users/tyarc/github/lrn/src')

from lrn.grammar_lessons import get_grammar_lessons as get_1st_lessons, get_lesson_by_title as get_1st_lesson
from lrn.grammar_lessons_2nd import get_grammar_lessons as get_2nd_lessons, get_lesson_by_title as get_2nd_lesson
from lrn.grammar_lessons_3rd import get_grammar_lessons as get_3rd_lessons, get_lesson_by_title as get_3rd_lesson


LESSON_GETTERS = {
    1: get_1st_lessons,
    2: get_2nd_lessons,
    3: get_3rd_lessons,
}

LESSON_TITLE_GETTERS = {
    1: get_1st_lesson,
    2: get_2nd_lesson,
    3: get_3rd_lesson,
}


class GrammarWorkbook:
    def __init__(self, grade=2, full=False):
        self.grade = grade
        self.full = full
        self.lessons = LESSON_GETTERS.get(grade, lambda f: [])(full)
        self.get_lesson = LESSON_TITLE_GETTERS.get(grade, lambda t, f: None)

    def list_lessons(self):
        """List all lesson titles."""
        return [lesson["title"] for lesson in self.lessons]

    def get_concept(self, title):
        """Get the grammar concept for a lesson."""
        lesson = self.get_lesson(title, self.grade, self.full)
        return lesson["concept"] if lesson else None

    def generate_practice(self, lesson_title_or_idx):
        """Generate practice sentences for a lesson."""
        if isinstance(lesson_title_or_idx, int):
            if 0 <= lesson_title_or_idx < len(self.lessons):
                lesson = self.lessons[lesson_title_or_idx]
            else:
                return []
        else:
            lesson = self.get_lesson(lesson_title_or_idx, self.grade, self.full)
            if not lesson:
                return []

        return lesson.get("sentences", [])

    def generate_assessment(self, lesson_title_or_idx, num_questions=5):
        """Generate assessment questions for a lesson."""
        sentences = self.generate_practice(lesson_title_or_idx)
        if not sentences:
            return []

        assessment = []
        for i, sentence in enumerate(sentences[:num_questions]):
            lesson = None
            if isinstance(lesson_title_or_idx, int):
                if 0 <= lesson_title_or_idx < len(self.lessons):
                    lesson = self.lessons[lesson_title_or_idx]
            else:
                lesson = self.get_lesson(lesson_title_or_idx, self.grade, self.full)

            assessment.append({
                "question_num": i + 1,
                "sentence": sentence,
                "concept": lesson.get("concept", "") if lesson else "",
            })
        return assessment

    def generate_spiral_review(self, current_lesson_idx, num_prior=3):
        """Generate spiral review of prior lessons."""
        review = []
        start = max(0, current_lesson_idx - num_prior)
        for i in range(start, current_lesson_idx):
            lesson = self.lessons[i]
            review.append({
                "title": lesson["title"],
                "sample_sentence": lesson["sentences"][0],
                "concept": lesson["concept"],
            })
        return review

    def generate_full_assessment(self, questions_per_lesson=3):
        """Generate full assessment covering all lessons."""
        assessment = []
        for lesson_idx, lesson in enumerate(self.lessons):
            lesson_assessment = self.generate_assessment(lesson_idx, questions_per_lesson)
            for q in lesson_assessment:
                q["lesson_title"] = lesson["title"]
                q["lesson_idx"] = lesson_idx
            assessment.extend(lesson_assessment)
        return assessment

    def generate_baseline_assessment(self):
        """Generate assessment for the baseline 10/15 lessons."""
        baseline_count = 10 if len(self.lessons) >= 10 else len(self.lessons)
        assessment = []
        for i in range(baseline_count):
            lesson = self.lessons[i]
            lesson_assessment = self.generate_assessment(i, 5)
            for q in lesson_assessment:
                q["lesson_title"] = lesson["title"]
                q["lesson_idx"] = i
            assessment.extend(lesson_assessment)
        return assessment

    def get_lessons_by_concept(self, concept_keyword):
        """Find lessons related to a concept keyword."""
        matches = []
        for lesson in self.lessons:
            if concept_keyword.lower() in lesson["concept"].lower():
                matches.append({
                    "title": lesson["title"],
                    "concept": lesson["concept"],
                    "num_sentences": len(lesson.get("sentences", [])),
                })
        return matches

    def generate_grammar_report(self):
        """Generate a report of grammar coverage."""
        concepts = {}
        for lesson in self.lessons:
            words = lesson["concept"].lower().split()
            for word in words:
                if len(word) > 4:
                    if word not in concepts:
                        concepts[word] = 0
                    concepts[word] += 1

        return {
            "grade": self.grade,
            "num_lessons": len(self.lessons),
            "total_sentences": sum(len(l.get("sentences", [])) for l in self.lessons),
            "lesson_titles": [l["title"] for l in self.lessons],
            "key_concepts": list(set(" ".join(l["concept"] for l in self.lessons).split()))[:20],
        }


def get_grammar_workbook(grade, full=False):
    """Factory function to get a grammar workbook."""
    return GrammarWorkbook(grade, full)


def generate_practice_for_grade(grade, lesson_idx=None, full=False):
    """Generate practice sentences for a grade."""
    wb = GrammarWorkbook(grade, full)
    if lesson_idx is not None:
        return wb.generate_practice(lesson_idx)
    return [wb.generate_practice(i) for i in range(len(wb.lessons))]


def generate_assessment_for_grade(grade, full=False, baseline_only=True):
    """Generate assessment for a grade."""
    wb = GrammarWorkbook(grade, full)
    if baseline_only:
        return wb.generate_baseline_assessment()
    return wb.generate_full_assessment()


if __name__ == "__main__":
    import json

    print("=" * 60)
    print("GRAMMAR WORKBOOK GENERATOR")
    print("=" * 60)

    for grade in [1, 2, 3]:
        print(f"\n--- GRADE {grade} ---")
        wb = GrammarWorkbook(grade, full=False)
        report = wb.generate_grammar_report()
        print(f"Lessons: {report['num_lessons']}")
        print(f"Total sentences: {report['total_sentences']}")
        print(f"Baseline lessons: 10")
        print(f"Full lessons: {report['num_lessons']}")

        print("\nLesson titles:")
        for i, title in enumerate(report['lesson_titles'][:5]):
            print(f"  {i+1}. {title}")
        if len(report['lesson_titles']) > 5:
            print(f"  ... and {len(report['lesson_titles']) - 5} more")

    print("\n" + "=" * 60)
    print("SAMPLE ASSESSMENT: Grade 2, Lesson 1")
    print("=" * 60)
    wb = GrammarWorkbook(grade=2, full=False)
    assessment = wb.generate_assessment(0, num_questions=5)
    for q in assessment:
        print(f"\nQ{q['question_num']}: {q['sentence']}")
        print(f"   Concept: {q['concept']}")

    print("\n" + "=" * 60)
    print("SPIRAL REVIEW: Grade 2, Lesson 10")
    print("=" * 60)
    review = wb.generate_spiral_review(9, num_prior=3)
    for r in review:
        print(f"\nReview: {r['title']}")
        print(f"  {r['concept']}")
        print(f"  Example: {r['sample_sentence']}")