
"""
features : 

1. Take user's data 
    - via text
    - via pdf
2. Job description
3. Prebuilts prompt, Give Prompt
4. Preview, Download 
"""


from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

parser = StrOutputParser()

prompt = """
You are an AI assistant that generates professional resumes in LaTeX format. 
You will be given two inputs: 

1. User Data (raw resume details including education, skills, experience, and projects).  
2. Job Description (role requirements, desired qualifications, and skills).  

Your task:  
- Extract and highlight the most relevant parts of the user data that match the job description.  
- Organize the information into a clear, ATS-friendly LaTeX resume.  
- Remove unrelated or weak details that do not support the job role.  
- Ensure formatting is clean, consistent, and professional.  

Output ONLY a complete LaTeX resume code between ```latex and ```.  
Do not include explanations, markdown text, or commentary.  

### Example Input Format:
[USER_DATA]: 
<Name, Education, Skills, Experience, Projects...>

[JOB_DESCRIPTION]:
<Role, Responsibilities, Requirements...>

"""

user_data = ("Harsh M. is an AI & Machine Learning Developer (B.Tech CSE - Specialization: AI/ML) based in India, "
               "reachable at harsh@example.com and +91-XXXXXXXXXX, with profiles on GitHub (github.com/harsh) and LinkedIn (linkedin.com/in/harsh). "
               "He is pursuing a B.Tech in Computer Science & Engineering with specialization in AI/ML from Your University (Expected/Year). "
               "An aspiring AI/ML developer with hands-on experience in NLP, computer vision, full-stack web technologies, and production-ready engineering using Node.js, TypeScript, and AWS, "
               "he has a strong foundation in supervised and unsupervised learning, with project experience in NER, summarization, OCR, and real-time systems. "
               "His skills include programming languages such as Python, Java, TypeScript, and SQL; ML/CV/NLP tools like TensorFlow, PyTorch basics, OpenCV, and techniques in NER and summarization; "
               "web and backend technologies such as Node.js, Express, Next.js, and React; datastores like PostgreSQL, AWS S3, and RDS; "
               "and tools including Git, Docker, AWS (S3, EC2, RDS basics), and LaTeX. He also has knowledge of RTK/Redux basics and recommender system concepts. "
               "Key projects include: a NER-based News Veracity Analysis using the Politifact dataset for correlation and fake/real news detection with feature engineering and predictive models; "
               "a Hospital Management & Doctor Availability System using Next.js frontend, Node.js backend, Google Maps, ABHA, and a CV module for real-time availability detection; "
               "a Real-time Quiz Platform built with React, WebSockets, and Google Auth for a multi-page SPA with performance-focused UX; "
               "and Text Summarization & OCR projects implementing summarization pipelines and OCR-based data extraction in Python. "
               "He has also conducted workshops for 72 students on website creation, CI/CD, Docker, and cloud fundamentals, "
               "and developed multiple student-focused web/MCQ study platforms and e-commerce features. He is seeking an internship or entry-level role in Machine Learning or AI engineering.")
job_description = ("We are seeking a passionate AI/ML Engineer Intern to join our team. "
                   "The role involves working on Natural Language Processing (NER, text summarization), "
                   "Computer Vision (OCR, real-time detection), and building production-ready AI systems. "
                   "You will contribute to model development, feature engineering, and deployment pipelines using Python, TensorFlow, and PyTorch basics. "
                   "The position also requires integration of AI modules into full-stack applications with Node.js, React, and cloud services (AWS S3, EC2, RDS). "
                   "Familiarity with SQL/PostgreSQL, Docker, and Git is expected. "
                   "The ideal candidate should have strong problem-solving skills, hands-on project experience, "
                   "and the ability to collaborate in building scalable AI-driven applications. "
                   "This internship offers exposure to real-world AI/ML workflows, production engineering, and modern cloud-based deployment practices.")


llm = ChatOpenAI(
    base_url = "http://127.0.0.1:1234/v1",
    model = "liquid/lfm2-1.2b",
    api_key = "not-needed",
    temperature = 0.3
)
template = PromptTemplate(
    template = prompt,
    input_variables = ["USER_DATA", "JOB_DESCRIPTION"],
    validate_template = True
)


chain = template | llm | parser


value = chain.invoke({
    "USER_DATA" : user_data,
    "JOB_DESCRIPTION" : job_description
})

print(value)