import sqlite3
import re
from openai import OpenAI
import os
from dotenv import load_dotenv
from datetime import datetime
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from PyPDF2 import PdfReader

load_dotenv()

class Industry40AssessmentAgent:
    def __init__(self, pdf_path):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.conn = sqlite3.connect('industry40_assessment_test.db')
        self.cursor = self.conn.cursor()
        self.embeddings = OpenAIEmbeddings()
        self.vector_store = None

    # Initialize knowledge base
        self.knowledge_base = self.process_pdf(pdf_path)
        print("Knowledge base initialized successfully!")

    def process_pdf(self, pdf_path):
        """Extract and process content from PDF"""
        # Read PDF
        reader = PdfReader(pdf_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text()

    def load_impuls_knowledge(self, text_content):
        """Load and process IMPULS documentation"""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        chunks = text_splitter.split_text(text_content)
        
        self.vector_store = Chroma.from_texts(
            chunks, 
            self.embeddings,
            persist_directory="./impuls"
        )
        print("IMPULS knowledge base loaded successfully!")

    def get_relevant_context(self, question, dimension, subdimension):
        """Get relevant IMPULS context"""
        if not self.vector_store:
            return ""
            
        search_query = f"Industry 4.0 standards for {dimension}, {subdimension}: {question}"
        docs = self.vector_store.similarity_search(search_query, k=2)
        return "\n".join([doc.page_content for doc in docs]) 

                        ######################################

    def _parse_analysis(self, content):
        """Process a single question response with IMPULS context"""
        lines = content.split('\n')
        result = {
            'score': 0,
            'justification': '',
            'recommendation': ''
        }
        
        for line in lines:
            if line.lower().startswith('score:'):
                try:
                    # Extract number from score line and convert to float
                    score_text = line.split(':')[1].strip()
                    # Handle potential range values like "4-5" by taking the average
                    if '-' in score_text:
                        scores = [float(x) for x in score_text.split('-')]
                        result['score'] = sum(scores) / len(scores)
                    else:
                        result['score'] = float(score_text)
                except:
                    result['score'] = 0
            elif line.lower().startswith('justification:'):
                result['justification'] = line.split(':', 1)[1].strip()
            elif line.lower().startswith('recommendation:'):
                result['recommendation'] = line.split(':', 1)[1].strip()
        
        return result

    def get_company_info(self):
        """Get available options from database and collect company information"""
        # Get available options from database
        self.cursor.execute("SELECT size_id, employee_range, revenue_range FROM company_sizes")
        company_sizes = self.cursor.fetchall()
        
        self.cursor.execute("SELECT name FROM industry_types")
        industry_types = [row[0] for row in self.cursor.fetchall()]
        
        self.cursor.execute("SELECT name FROM manufacturing_types")
        manufacturing_types = [row[0] for row in self.cursor.fetchall()]

        # Display options and get input
        print("\nCompany Information:")
        print("-------------------")
        print("\nAvailable company sizes:")
        for size_id, employees, revenue in company_sizes:
            print(f"{size_id.title()}: {employees} employees, Revenue: {revenue}")

        while True:
            size = input("\nEnter company size: ").lower()
            if size in [s[0] for s in company_sizes]:
                break
            print("Invalid size. Please choose from the options above.")

        print("\nAvailable industry types:")
        print(", ".join(industry_types))
        while True:
            industry = input("\nEnter industry type: ")
            if industry in industry_types:
                break
            print("Invalid industry. Please choose from the options above.")

        print("\nAvailable manufacturing types:")
        print(", ".join(manufacturing_types))
        while True:
            mfg_type = input("\nEnter manufacturing type: ")
            if mfg_type in manufacturing_types:
                break
            print("Invalid type. Please choose from the options above.")

        location = input("\nCompany location (country): ")

        return {
            'size': size,
            'industry_type': industry,
            'manufacturing_type': mfg_type,
            'location': location
        }

    def clean_response(self, response, response_type):
        """Clean and validate user response"""
        try:
            # Remove any special characters except digits, dots, and percentage signs
            cleaned = re.sub(r'[^0-9.%]', '', str(response))
            
            if response_type == 'percentage':
                # Remove percentage sign if present
                cleaned = cleaned.replace('%', '')
                value = float(cleaned)
                if value < 0 or value > 100:
                    raise ValueError("Percentage must be between 0 and 100")
                return str(value)
            else:  # number type
                value = float(cleaned)
                if value < 0:
                    raise ValueError("Number must be positive")
                return str(value)
        except ValueError as e:
            raise ValueError(f"Invalid input: {str(e)}")

    def get_dimensions(self):
        """Get all dimensions from database"""
        self.cursor.execute("SELECT dimension_id, name FROM dimensions")
        return self.cursor.fetchall()

    def get_subdimensions(self, dimension_id):
        """Get subdimensions for a specific dimension"""
        self.cursor.execute("""
            SELECT subdimension_id, name 
            FROM subdimensions 
            WHERE dimension_id = ?
        """, (dimension_id,))
        return self.cursor.fetchall()

    def get_questions(self, subdimension_id):
        """Get questions for a specific subdimension"""
        self.cursor.execute("""
            SELECT question_id, question_text, response_type, unit 
            FROM questions 
            WHERE subdimension_id = ?
        """, (subdimension_id,))
        return self.cursor.fetchall()

    def process_response(self, session_id, question_id, response, company_context):
        """Process a single question response with IMPULS context"""
        # Get question and context details
        self.cursor.execute("""
            SELECT 
                q.question_text, 
                q.response_type, 
                q.unit,
                d.name as dimension_name,
                s.name as subdimension_name
            FROM questions q
            JOIN dimensions d ON q.dimension_id = d.dimension_id
            JOIN subdimensions s ON q.subdimension_id = s.subdimension_id
            WHERE q.question_id = ?
        """, (question_id,))
        
        question_data = dict(zip(
            ['question_text', 'response_type', 'unit', 'dimension_name', 'subdimension_name'],
            self.cursor.fetchone()
        ))

        # Clean and validate response
        cleaned_response = self.clean_response(response, question_data['response_type'])

        # Create analysis prompt
        prompt = f"""
        Context:
        - Company Size: {company_context['size']}
        - Industry: {company_context['industry_type']}
        - Manufacturing: {company_context['manufacturing_type']}
        - Location: {company_context['location']}
        
        Question: {question_data['question_text']}
        Response: {cleaned_response} {question_data.get('unit', '')}
        
        From {question_data['dimension_name']} dimension, {question_data['subdimension_name']} subdimension.
        
        Based on IMPULS Industry 4.0 standards - from the PDF - and considering the company's context:       
        1. Score (0-5, where 0=Newcomer, 2.5=Learner, 5=Leader)
        2. Brief, contextualized justification based on IMPULS criteria
        3. Practical recommendation for their size/industry (take insight from the knwoledge base 'impulss.pdf')
        4. If the input of the user is not clear don't generate creative answers.
        
        Format: 
        Score: [number]
        Justification: [text]
        Recommendation: [text]
        """

        # Get AI analysis
        analysis = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are an Industry 4.0 expert assessment system. You have a professional tone, and generate quantifiable and practical responses."},
                {"role": "user", "content": prompt}
            ]
        )

        # Parse and store response
        result = self._parse_analysis(analysis.choices[0].message.content)
        
        self.cursor.execute("""
            INSERT INTO question_responses 
            (session_id, question_id, response_value, question_score, feedback, ai_analysis)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            session_id,
            question_id,
            float(cleaned_response),
            result['score'],
            result['justification'],
            result['recommendation']
        ))
        self.conn.commit()

        return result

    # Add these methods to your Industry40AssessmentAgent class:

    def calculate_dimension_score(self, session_id, dimension_id):
        """Calculate average score for a dimension"""
        try:
            self.cursor.execute("""
                SELECT AVG(qr.question_score)
                FROM question_responses qr
                JOIN questions q ON qr.question_id = q.question_id
                WHERE qr.session_id = ? AND q.dimension_id = ?
            """, (session_id, dimension_id))
            
            score = self.cursor.fetchone()[0]
            return round(score, 2) if score is not None else 0
        except Exception as e:
            print(f"Error calculating dimension score: {e}")
            return 0

    def calculate_overall_score(self, session_id):
        """Calculate overall assessment score"""
        try:
            self.cursor.execute("""
                SELECT AVG(question_score)
                FROM question_responses
                WHERE session_id = ?
            """, (session_id,))
            
            score = self.cursor.fetchone()[0]
            return round(score, 2) if score is not None else 0
        except Exception as e:
            print(f"Error calculating overall score: {e}")
            return 0

    def get_dimension_summary(self, session_id):
        """Get summary of all dimension scores"""
        scores = {}
        try:
            self.cursor.execute("""
                SELECT 
                    d.name,
                    AVG(qr.question_score) as avg_score
                FROM dimensions d
                JOIN questions q ON d.dimension_id = q.dimension_id
                JOIN question_responses qr ON q.question_id = qr.question_id
                WHERE qr.session_id = ?
                GROUP BY d.dimension_id, d.name
            """, (session_id,))
            
            for row in self.cursor.fetchall():
                scores[row[0]] = round(row[1], 2) if row[1] is not None else 0
                
        except Exception as e:
            print(f"Error getting dimension summary: {e}")
        
        return scores

def run_assessment(pdf_path):
    """Run assessment with PDF knowledge base"""
    # Initialize agent with PDF
    print("Initializing assessment agent...")
    agent = Industry40AssessmentAgent(pdf_path)
    
    # Get company info
    print("\nWelcome to Industry 4.0 Maturity Assessment")
    company_name = input("Company name: ")
    company_context = agent.get_company_info()
    
    # Start assessment session
    agent.cursor.execute("""
        INSERT INTO assessment_sessions 
        (company_name, company_size, industry_type, manufacturing_type, location) 
        VALUES (?, ?, ?, ?, ?)
    """, (
        company_name,
        company_context['size'],
        company_context['industry_type'],
        company_context['manufacturing_type'],
        company_context['location']
    ))
    agent.conn.commit()
    session_id = agent.cursor.lastrowid

    # Run assessment
    dimensions = agent.get_dimensions()
    dimension_scores = {}

    for dim_id, dim_name in dimensions:
        print(f"\n=== {dim_name} ===")
        subdimensions = agent.get_subdimensions(dim_id)
        
        for subdim_id, subdim_name in subdimensions:
            print(f"\n-- {subdim_name} --")
            questions = agent.get_questions(subdim_id)
            
            for q_id, q_text, q_type, q_unit in questions:
                while True:
                    print(f"\n{q_text}")
                    if q_unit:
                        print(f"(Answer in {q_unit})")
                    
                    response = input("Your answer: ")
                    try:
                        result = agent.process_response(session_id, q_id, response, company_context)
                        print(f"\nScore: {result['score']}/5")
                        print(f"Analysis: {result['justification']}")
                        print(f"Recommendation: {result['recommendation']}")
                        break
                    except ValueError as e:
                        print(f"Error: {str(e)}. Please try again.")
        
        # Calculate dimension score
        dimension_scores[dim_name] = agent.calculate_dimension_score(session_id, dim_id)
        print(f"\n{dim_name} Dimension Score: {dimension_scores[dim_name]}/5")

    # Show final summary
    print("\n=== Assessment Summary ===")
    print(f"Company: {company_name}")
    print(f"Size: {company_context['size']}")
    print(f"Industry: {company_context['industry_type']}")
    print("\nDimension Scores:")
    for dim_name, score in dimension_scores.items():
        print(f"{dim_name}: {score}/5")
    
    overall_score = agent.calculate_overall_score(session_id)
    print(f"\nOverall Industry 4.0 Maturity Score: {overall_score}/5")
    
    print("\nAssessment completed!")

if __name__ == "__main__":
    pdf_path = "impulss.pdf"  # Path to your IMPULS PDF
    run_assessment(pdf_path)