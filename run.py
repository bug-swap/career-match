from src.models.entity_extractor import EntityExtractorTrainer
from src.models.entity_extractor.model import EntityExtractor

trainer = EntityExtractorTrainer()
trainer.train()
trainer.test()

extractor = EntityExtractor()

result = extractor.extract("""
    John Doe
    john@email.com | (555) 123-4567
    Software Engineer at Google, 2020-Present
    MIT, Bachelor of Science
""")

print(result.name)        # "John Doe"
print(result.email)       # "john@email.com"
print(result.companies)   # ["Google"]
print(result.job_titles)  # ["Software Engineer"]
