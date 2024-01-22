
import requests
import pandas as pd


url = 'http://localhost:9696/predict'

job ={'job_id': 4351,
  'title': 'Assembly & Test Engineer',
  'location': 'US, TX, Houston',
  'department': 'Engineering',
  'salary_range': None,
  'company_profile': None,
  'description': "Corporate overviewAker Solutions is a global provider of products, systems and services to the oil and gas industry. Our engineering, design and technology bring discoveries into production and maximize recovery from each petroleum field. We employ approximately 28,000 people in about 30 countries. Go to #URL_0fa3f7c5e23a16de16a841e368006cae916884407d90b154dfef3976483a71ae# for more information on our business, people and values.We are looking for individuals who are prepared to take a position. Not only a position within Aker Solutions, but also a position on the exciting challenges the global oil and gas industry faces now and in the future.Aker Solutions' Subsea team based in Houston, TX is responsible for design, engineering, procurement and assembly/test of complex subsea systems. We need new talents who can strengthen our team and support our ambitious growth plans within the subsea market. We are looking for: Assembly &amp; Test Engineer.Responsibilities and tasks • Assembly and Test Engineer initiates work tasks to be carried out by the workshop, follows the quality standards and plans the use of resources in order to increase the efficiency of the organization• Responsible for delivering on time according to plans and manages the day-to-day work tasks in a cost effective, safe, and efficient manner• Create work orders and issue work packages in accordance with contract/project requirements and according to procedure• Arrange and call for pre-job meeting with work shop personnel • Participate on hand-over (tool box) meetings in work shop• Responsible for updating work packages related to specific project needsor NCR (CQN) reports• Follow-up progress in workshop and respond without delay on requests for support• Making requisitions towards supply chain to handle unplanned activities• Inform project manager or line manager when Variation Orders is required, for additional work on existing SOW or when change in original SOW• Continuously verify that all relevant documents are completed according to job package procedure • Evaluate the risk of operations and the compliance with laws, standards and policies• Participates in SAFOP/SJA and other HSE related activities in WS• Make sure that special tools and equipment to the planned work (according to work order/package) in assigned projects are available and certified • Control/monitor that necessary materials are received and in place before start of assembly and test• Contribute to the design and improvement of special tools to perform the work in the workshop• Inform and if applicable submit formal status reports to project manager on assigned work tasks/SOW • Write NCRs when deviation is detected• Provide Project manager with input to lessons learned• Participate and contribute with expertise in Tender work",
  'requirements': 'Qualifications &amp; personal attributes • Mechanical/technical experience and insight, engineering degree or relevant business experience• Preferably 3-5 years’ experience in Subsea Workshop or similar industry• Structured and methodical• Strong interpersonal skills with the ability to work effectively both within a team environment and with limited supervision, ability to take lead • SAP and Microsoft Office skills preferable• Fluent in English with good verbal and written communication skillsCompany values:• Ensure understanding of HSE standards, model HSE behaviours, minimize accidents• Ensure team complies with policies and procedures• Create and build cohesive teamwork• Enhance client satisfaction on all products, service and relationship with company',
  'benefits': 'We offer • Friendly colleagues in an industry with a bright future.• An environment where you are encouraged to develop your skills and share your knowledge with your colleagues.• Competitive benefits and strong focus on work-life balance.',
  'telecommuting': 0,
  'has_company_logo': 0,
  'has_questions': 0,
  'employment_type': None,
  'required_experience': None,
  'required_education': None,
  'industry': 'Oil & Energy',
  'function': 'Engineering'}

job_id = job['job_id']
job_title = job['title']
job_location = job['location']


response = requests.post(url, json=job).json()

if response['job'] == True:
    print(f'The job with id {job_id}/{job_title} in {job_location} seems to be a fraud')
else:
    print(f'The job with id {job_id}/{job_title} in {job_location} seems to be legit')


