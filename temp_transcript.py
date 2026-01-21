data = {
  "students": [
    {
      "id": "S001",
      "name": "Alice",
      "semesters": [
        {
          "term": "Fall 2023",
          "subjects": [
            { "name": "Math", "credits": 4, "performance": { "assignments": 80, "exams": 70, "attendance": 85 } },
            { "name": "Physics", "credits": 3, "performance": { "assignments": 90, "exams": 60, "attendance": 70 } }
          ]
        }
      ]
    },
    {
      "id": "S002",
      "name": "Bob",
      "semesters": [
        {
          "term": "Fall 2023",
          "subjects": [
            { "name": "Math", "credits": 4, "performance": { "assignments": 85, "exams": 75, "attendance": 90 } },
            { "name": "English", "credits": 2, "performance": { "assignments": 95, "exams": 82, "attendance": 60 } }
          ]
        }
      ]
    }
  ]
}

def calculate_final_grade(performance):
    return 0.3 * performance['assignments'] + 0.5 * performance['exams'] + 0.2 * performance['attendance']

def calculate_gpa(subjects):
    total_weighted_grades = 0
    total_credits = 0
    for subject in subjects:
        final_grade = calculate_final_grade(subject['performance'])
        subject['final_grade'] = final_grade
        total_weighted_grades += final_grade * subject['credits']
        total_credits += subject['credits']
    gpa = (total_weighted_grades / total_credits) / 100 * 4
    return gpa, total_credits

def generate_transcript(student):
    print(f"Transcript for {student['name']} (ID: {student['id']})")
    cumulative_weighted_grades = 0
    cumulative_credits = 0
    for semester in student['semesters']:
        print(f"Term: {semester['term']}")
        gpa, credits = calculate_gpa(semester['subjects'])
        cumulative_weighted_grades += sum(subj['final_grade'] * subj['credits'] for subj in semester['subjects'])
        cumulative_credits += credits
        honors = " with Honors" if gpa >= 3.7 else ""
        print(f"GPA for this semester: {gpa:.2f}{honors}")
    cumulative_gpa = (cumulative_weighted_grades / cumulative_credits) / 100 * 4
    honors = " with Honors" if cumulative_gpa >= 3.7 else ""
    print(f"Cumulative GPA: {cumulative_gpa:.2f}{honors}\n")

for student in data['students']:
    generate_transcript(student)
