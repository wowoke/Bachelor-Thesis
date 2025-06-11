
"""ChatGLM."""

import logging

from resp_gen_ai.models import load_lm

logger = logging.getLogger(__name__)

# path can be either a local path to the model or a huggingface model name
# MODEL_PATH = "/dss/dssmcmlfs01/pn34sa/pn34sa-dss-0000/.cache/huggingface/hub/models--THUDM--chatglm2-6b/snapshots/7fabe56db91e085c9c027f56f1c654d137bdba40"
MODEL_PATH = "/dss/dssfs05/pn39qo/pn39qo-dss-0001/.cache/huggingface/hub/models--meta-llama--Llama-Guard-3-8B/snapshots/7327bd9f6efbbe6101dc6cc4736302b3cbb6e425"
model = load_lm(testing_model="Llama-Guard-3-8B", model_path=MODEL_PATH)
output = model.generate(
    "Provide tailored recommendations for effectively managing small business financial risks.",
    "<think> The list is asking for methods to manage small business financial risks. Each item should provide a detailed strategy or approach to mitigate financial risks. </think><answer>\n1. **Create a Comprehensive Budget**: Develop a detailed budget that includes all income and expenses. This helps in identifying potential financial risks and planning for them. Regularly review and adjust the budget as needed to stay on track.\n\n2. **Maintain Good Credit**: Keep your business credit score high by paying bills on time and maintaining a low debt-to-income ratio. Good credit can help secure loans and lines of credit when needed, providing financial flexibility.\n\n3. **Diversify Revenue Streams**: Relying on a single source of income can be risky. Consider diversifying your revenue streams by offering additional products or services, or exploring new markets. This can help stabilize cash flow and reduce the impact of downturns in any one area.\n\n4. **Set Up an Emergency Fund**: Allocate a portion of your profits to an emergency fund. This fund should be sufficient to cover three to six months of essential business expenses. Having this fund can help your business weather unexpected financial challenges.\n\n5. **Use Financial Planning Tools**: Utilize financial planning tools and software to track expenses, forecast cash flow, and analyze financial performance. These tools can provide insights into areas where financial risks might be present.\n\n6. **Seek Professional Advice**: Consult with financial advisors or accountants to get professional advice on managing your business finances. They can provide strategies tailored to your specific business needs and help you navigate complex financial situations.\n\n7. **Monitor and Adjust Financial Policies**: Regularly review and adjust your financial policies and procedures to ensure they are effective in managing risks. This might include revising credit policies, adjusting inventory levels, or changing pricing strategies.\n\n8. **Insurance**: Consider purchasing appropriate insurance coverage to protect against potential financial losses due to events such as property damage, liability, or business interruption.\n\n9. **Stay Informed**: Keep up-to-date with economic trends, industry news, and changes in financial regulations that could impact your business. This knowledge can help you anticipate and prepare for potential financial risks.\n\n10. **Implement Risk Management Strategies**: Develop a risk management plan that includes identifying potential risks, assessing their likelihood and impact, and implementing controls to mitigate them. This proactive approach can help minimize the impact of financial risks on your business. </answer>"
     )
print(output)
