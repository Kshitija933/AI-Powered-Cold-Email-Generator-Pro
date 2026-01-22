import pandas as pd
import json
import random
from datetime import datetime

class AdvancedDatasetGenerator:
    """Advanced dataset generator with expanded templates and variations"""
    
    def __init__(self):
        self.industries = [
            "Technology", "Healthcare", "Finance", "Education", "Retail",
            "Manufacturing", "Real Estate", "Marketing", "Logistics", "SaaS"
        ]
        
        self.pain_points = {
            "Technology": ["scalability issues", "legacy system integration", "security vulnerabilities"],
            "Healthcare": ["patient data management", "compliance requirements", "operational efficiency"],
            "Finance": ["regulatory compliance", "risk management", "transaction processing speed"],
            "Education": ["student engagement", "administrative overhead", "learning outcomes"],
            "Retail": ["inventory management", "customer retention", "omnichannel experience"],
            "Manufacturing": ["supply chain visibility", "quality control", "downtime reduction"],
            "Real Estate": ["lead generation", "property management", "market analytics"],
            "Marketing": ["ROI tracking", "campaign performance", "customer acquisition cost"],
            "Logistics": ["route optimization", "delivery tracking", "cost reduction"],
            "SaaS": ["customer churn", "product adoption", "scaling infrastructure"]
        }
        
        self.value_propositions = [
            "reduce costs by {percent}%",
            "improve efficiency by {percent}%",
            "increase revenue by {percent}%",
            "save {hours} hours per week",
            "accelerate time-to-market by {percent}%",
            "enhance customer satisfaction by {percent}%",
            "automate {percent}% of manual processes",
            "reduce errors by {percent}%"
        ]
        
    def generate_comprehensive_dataset(self, num_samples=50):
        """Generate a comprehensive dataset with variations"""
        dataset = []
        
        # Core templates with advanced structures
        templates = self._create_advanced_templates()
        
        # Generate samples
        for i in range(num_samples):
            template = random.choice(templates)
            sample = self._personalize_template(template, i)
            dataset.append(sample)
        
        return dataset
    
    def _create_advanced_templates(self):
        """Create advanced email templates with different structures"""
        return [
            {
                "structure": "problem_solution",
                "tone": "professional",
                "template": """Subject: Solving {pain_point} at {company}

Dear {name},

I noticed that {company} has been experiencing challenges with {pain_point}. This is a common issue in the {industry} sector that can significantly impact {business_metric}.

Our {product} specifically addresses this by:
• {benefit_1}
• {benefit_2}
• {benefit_3}

Companies like {case_study_company} have seen {value_prop} after implementing our solution.

Would you be available for a 20-minute discovery call next week to discuss how we can help {company} achieve similar results?

Best regards,
{sender_name}"""
            },
            {
                "structure": "social_proof",
                "tone": "friendly",
                "template": """Subject: How {case_study_company} achieved {result} 🚀

Hi {name},

I hope you're having a great week! I wanted to reach out because I've been following {company}'s growth, and I'm impressed by your recent {achievement}.

I thought you might be interested in how {case_study_company}, another leader in {industry}, used our {product} to {value_prop}.

Here's what they experienced:
✨ {benefit_1}
✨ {benefit_2}
✨ {benefit_3}

I'd love to share more details and explore if there's a fit for {company}. Are you free for a quick chat this week?

Cheers,
{sender_name}"""
            },
            {
                "structure": "data_driven",
                "tone": "technical",
                "template": """Subject: {company}'s {metric} Optimization Analysis

Dear {name},

Based on industry benchmarks for {industry} companies, organizations typically lose ${cost} annually due to {pain_point}.

Our {product} leverages {technology} to:
→ {benefit_1}
→ {benefit_2}
→ {benefit_3}

Technical Specifications:
• Integration: {integration_detail}
• Deployment: {deployment_detail}
• Security: {security_detail}

ROI Analysis: Most clients see {value_prop} within {timeframe}.

I'd appreciate the opportunity to present a customized analysis for {company}.

Best regards,
{sender_name}"""
            },
            {
                "structure": "storytelling",
                "tone": "creative",
                "template": """Subject: A story about {industry} transformation 📖

Hey {name},

Let me tell you a quick story...

Last year, {case_study_company} was struggling with {pain_point}. Their {role} reached out to us, frustrated and looking for answers.

Fast forward 6 months: They've {value_prop}, their team is happier, and they're scaling faster than ever.

The secret? Our {product} that offers:
💡 {benefit_1}
🎯 {benefit_2}
🚀 {benefit_3}

Your situation at {company} reminds me of their journey. Want to write your own success story?

Let's connect!
{sender_name}"""
            },
            {
                "structure": "question_based",
                "tone": "consultative",
                "template": """Subject: Quick question about {company}'s {metric}

Hi {name},

I have a question for you:

How much time does your team currently spend on {pain_point}?

Most {industry} companies we work with report spending {hours} hours weekly on this alone. That's {cost} in productivity costs annually.

Our {product} helps teams like yours:
→ {benefit_1}
→ {benefit_2}
→ {benefit_3}

Result: {value_prop}

Would it be worth 15 minutes to explore how {company} could achieve similar results?

Looking forward to your thoughts,
{sender_name}"""
            },
            {
                "structure": "comparison",
                "tone": "professional",
                "template": """Subject: {company} vs. Industry Leaders in {metric}

Dear {name},

Industry analysis shows that top-performing {industry} companies are {value_prop} compared to their competitors.

The key differentiator? Advanced {product} solutions.

Where {company} stands to gain:
1. {benefit_1}
2. {benefit_2}
3. {benefit_3}

Current Market Leaders Using Our Platform:
• {case_study_company}
• {competitor_1}
• {competitor_2}

Let's discuss how {company} can join this competitive advantage.

Best regards,
{sender_name}"""
            },
            {
                "structure": "urgency_based",
                "tone": "professional",
                "template": """Subject: Time-Sensitive: {industry} Compliance Update

Dear {name},

With the upcoming {compliance_event} deadline approaching, {industry} companies are facing increased pressure around {pain_point}.

Our {product} ensures:
✓ {benefit_1}
✓ {benefit_2}
✓ {benefit_3}

Quick Implementation: {timeframe} from contract to deployment.

Given the timeline, I wanted to reach out now to ensure {company} has adequate time to evaluate and implement a solution.

Can we schedule a brief call this week?

Best regards,
{sender_name}"""
            },
            {
                "structure": "personalized_insight",
                "tone": "friendly",
                "template": """Subject: Loved your recent {content_type} about {topic} 💡

Hi {name},

Your {content_type} on {topic} really resonated with me, especially your point about {specific_point}.

It actually ties perfectly into what we're doing at {sender_company}. Our {product} helps {industry} leaders like {company} tackle {pain_point} by:

🎯 {benefit_1}
🎯 {benefit_2}
🎯 {benefit_3}

Result: {value_prop}

Given your insights on {topic}, I think you'd find our approach interesting. Want to exchange ideas over a quick call?

Best,
{sender_name}"""
            }
        ]
    
    def _personalize_template(self, template_data, index):
        """Personalize template with realistic data"""
        
        # Generate realistic data
        names = ["Sarah Johnson", "Michael Chen", "Emily Rodriguez", "David Park", 
                 "Lisa Wang", "Robert Taylor", "Amanda Foster", "James Wilson",
                 "Nicole Martinez", "Alex Thompson", "Jessica Lee", "Kevin Brown"]
        
        companies = ["TechFlow Solutions", "InnovateHealth", "FinanceFirst", 
                    "EduTech Pro", "RetailGenius", "ManufactureMax", "RealEstate360",
                    "MarketingPro", "LogisticsHub", "CloudScale", "DataDrive", "AICore"]
        
        roles = ["CEO", "CTO", "VP of Engineering", "Head of Operations", 
                "Chief Data Officer", "VP of Sales", "Director of Technology",
                "VP of Product", "Chief Marketing Officer", "Head of Innovation"]
        
        products = ["AI Analytics Platform", "Cloud Infrastructure Solution",
                   "Workflow Automation System", "Data Integration Platform",
                   "Security Compliance Suite", "Customer Intelligence Tool",
                   "Predictive Analytics Engine", "Process Optimization Platform"]
        
        # Select random data
        industry = random.choice(self.industries)
        name = random.choice(names)
        company = random.choice(companies)
        role = random.choice(roles)
        product = random.choice(products)
        
        # Generate metrics
        percent = random.randint(30, 70)
        hours = random.randint(10, 30)
        cost = random.randint(50, 500) * 1000
        
        # Get industry-specific pain points
        pain_point = random.choice(self.pain_points.get(industry, ["operational challenges"]))
        
        # Generate value proposition
        value_prop_template = random.choice(self.value_propositions)
        value_prop = value_prop_template.format(percent=percent, hours=hours)
        
        # Benefits
        benefits = [
            f"Automated {pain_point} processes",
            f"Real-time monitoring and alerts",
            f"Seamless integration with existing systems",
            f"Advanced analytics and reporting",
            f"Scalable cloud-based architecture",
            f"Enterprise-grade security and compliance",
            f"24/7 customer support",
            f"Customizable workflows and dashboards"
        ]
        
        # Fill template
        email_content = template_data["template"].format(
            name=name.split()[0],
            company=company,
            industry=industry,
            pain_point=pain_point,
            product=product,
            value_prop=value_prop,
            benefit_1=random.choice(benefits),
            benefit_2=random.choice([b for b in benefits if b != benefits[0]]),
            benefit_3=random.choice([b for b in benefits if b not in benefits[:2]]),
            case_study_company=random.choice(companies),
            business_metric="operational efficiency",
            achievement="product launch",
            result=f"{percent}% improvement",
            metric="efficiency",
            cost=f"{cost:,}",
            hours=hours,
            technology="machine learning and automation",
            integration_detail="REST API, webhooks, native connectors",
            deployment_detail="Cloud or on-premise, 48-hour setup",
            security_detail="SOC2, GDPR, ISO 27001 compliant",
            timeframe="3-6 months",
            role=role,
            competitor_1=random.choice(companies),
            competitor_2=random.choice(companies),
            compliance_event="regulatory compliance",
            content_type="article",
            topic="digital transformation",
            specific_point="customer-centric approach",
            sender_company="Growth Solutions Inc",
            sender_name="Alex Morgan"
        )
        
        return {
            "recipient_name": name,
            "company": company,
            "role": role,
            "industry": industry,
            "product": product,
            "tone": template_data["tone"],
            "structure": template_data["structure"],
            "pain_point": pain_point,
            "value_proposition": value_prop,
            "email": email_content
        }

# Main execution
if __name__ == "__main__":
    print("=" * 70)
    print("ADVANCED COLD EMAIL DATASET GENERATOR")
    print("=" * 70)
    
    generator = AdvancedDatasetGenerator()
    
    # Generate comprehensive dataset
    print("\n🔄 Generating advanced dataset with 50 diverse samples...")
    dataset = generator.generate_comprehensive_dataset(num_samples=50)
    
    # Create DataFrame
    df = pd.DataFrame(dataset)
    
    # Save to multiple formats
    df.to_csv('cold_email_dataset.csv', index=False)
    df.to_json('cold_email_dataset.json', orient='records', indent=2)
    
    # Save metadata
    metadata = {
        "generated_at": datetime.now().isoformat(),
        "total_samples": len(dataset),
        "industries": generator.industries,
        "tones": list(set(d['tone'] for d in dataset)),
        "structures": list(set(d['structure'] for d in dataset)),
        "statistics": {
            "avg_email_length": df['email'].str.len().mean(),
            "avg_word_count": df['email'].str.split().str.len().mean(),
            "unique_companies": df['company'].nunique(),
            "unique_roles": df['role'].nunique()
        }
    }
    
    with open('dataset_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Display statistics
    print(f"\n✅ Dataset generated successfully!")
    print(f"📊 Total samples: {len(dataset)}")
    print(f"🏢 Unique companies: {df['company'].nunique()}")
    print(f"🎭 Email tones: {', '.join(df['tone'].unique())}")
    print(f"📝 Email structures: {', '.join(df['structure'].unique())}")
    print(f"🏭 Industries covered: {len(generator.industries)}")
    
    print("\n📄 Files created:")
    print("   • cold_email_dataset.csv")
    print("   • cold_email_dataset.json")
    print("   • dataset_metadata.json")
    
    print("\n📋 Sample distribution:")
    print(df.groupby(['tone', 'structure']).size().to_string())
    
    print("\n" + "=" * 70)
    print("Dataset ready for training!")
    print("=" * 70)