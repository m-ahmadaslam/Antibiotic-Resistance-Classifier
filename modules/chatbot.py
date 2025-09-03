"""
Chatbot Page Module
===================

Contains the AI chatbot functionality for the healthcare application.
"""

import streamlit as st

def chatbot_page():
    """AI Chatbot page for user queries"""
    st.header("🤖 AI Healthcare Assistant")
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Chat interface
    st.subheader("💬 Ask me about antibiotics and drug-bug matches!")
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask a question..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate AI response
        with st.chat_message("assistant"):
            response = generate_chatbot_response(prompt)
            st.markdown(response)
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})
    
    # Quick action buttons
    st.subheader("🚀 Quick Actions")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("💊 Antibiotic Info", use_container_width=True):
            # Add user message
            st.session_state.messages.append({
                "role": "user", 
                "content": "Tell me about common antibiotics and their uses"
            })
            # Add AI response
            response = generate_chatbot_response("Tell me about common antibiotics and their uses")
            st.session_state.messages.append({
                "role": "assistant", 
                "content": response
            })
            st.rerun()
    
    with col2:
        if st.button("🦠 Organism Info", use_container_width=True):
            # Add user message
            st.session_state.messages.append({
                "role": "user", 
                "content": "What are the most common organisms in healthcare infections?"
            })
            # Add AI response
            response = generate_chatbot_response("What are the most common organisms in healthcare infections?")
            st.session_state.messages.append({
                "role": "assistant", 
                "content": response
            })
            st.rerun()
    
    with col3:
        if st.button("🔍 Check Match", use_container_width=True):
            # Add user message
            st.session_state.messages.append({
                "role": "user", 
                "content": "How do I check if an antibiotic matches an organism?"
            })
            # Add AI response
            response = generate_chatbot_response("How do I check if an antibiotic matches an organism?")
            st.session_state.messages.append({
                "role": "assistant", 
                "content": response
            })
            st.rerun()

def generate_chatbot_response(prompt):
    """Generate chatbot response based on user query"""
    prompt_lower = prompt.lower()
    
    # Antibiotic-related queries
    if any(word in prompt_lower for word in ['antibiotic', 'drug', 'medicine']):
        if 'ceftriaxone' in prompt_lower or 'cef' in prompt_lower:
            return """**Ceftriaxone** is a third-generation cephalosporin antibiotic.
            
**Uses:**
- Respiratory infections
- Urinary tract infections
- Meningitis
- Gonorrhea

**Common organisms it treats:**
- Escherichia coli
- Klebsiella pneumoniae
- Streptococcus pneumoniae
- Neisseria gonorrhoeae

**Match/Mismatch:** It's generally effective against Gram-negative bacteria but may not work against resistant strains."""
        
        elif 'meropenem' in prompt_lower or 'carbapenem' in prompt_lower:
            return """**Meropenem** is a carbapenem antibiotic, often used as a last resort.

**Uses:**
- Severe infections
- Hospital-acquired infections
- Multi-drug resistant organisms

**Common organisms it treats:**
- Pseudomonas aeruginosa
- Acinetobacter baumannii
- Extended-spectrum beta-lactamase (ESBL) producers

**Match/Mismatch:** Very broad spectrum, but resistance is increasing. Should be used carefully."""
        
        elif 'vancomycin' in prompt_lower:
            return """**Vancomycin** is a glycopeptide antibiotic.

**Uses:**
- MRSA infections
- Clostridium difficile (oral)
- Serious Gram-positive infections

**Common organisms it treats:**
- Staphylococcus aureus (including MRSA)
- Enterococcus species
- Clostridium difficile

**Match/Mismatch:** Only effective against Gram-positive bacteria. Won't work against Gram-negative organisms."""
        
        else:
            return """**Common Antibiotics and Their Uses:**

**Penicillins:**
- Amoxicillin: Respiratory, ear, and urinary tract infections
- Piperacillin: Hospital-acquired infections

**Cephalosporins:**
- Ceftriaxone: Respiratory, urinary, and bloodstream infections
- Cefepime: Severe infections, including Pseudomonas

**Carbapenems:**
- Meropenem: Last-resort antibiotic for severe infections
- Imipenem: Multi-drug resistant organisms

**Aminoglycosides:**
- Amikacin: Gram-negative infections
- Gentamicin: Urinary and bloodstream infections

**Fluoroquinolones:**
- Ciprofloxacin: Urinary and respiratory infections
- Levofloxacin: Respiratory infections

**Glycopeptides:**
- Vancomycin: MRSA and serious Gram-positive infections

**Match/Mismatch:** Always check susceptibility testing results before prescribing."""
    
    # Organism-related queries
    elif any(word in prompt_lower for word in ['organism', 'bacteria', 'infection', 'e.coli', 'klebsiella']):
        if 'e.coli' in prompt_lower or 'escherichia' in prompt_lower:
            return """**Escherichia coli (E. coli)** is a common Gram-negative bacterium.

**Common infections:**
- Urinary tract infections
- Gastrointestinal infections
- Bloodstream infections

**Effective antibiotics:**
- Ceftriaxone
- Ciprofloxacin
- Amikacin
- Piperacillin/tazobactam

**Resistance concerns:**
- ESBL-producing strains
- Carbapenem-resistant strains

**Match/Mismatch:** Depends on susceptibility testing results."""
        
        elif 'klebsiella' in prompt_lower:
            return """**Klebsiella pneumoniae** is a Gram-negative bacterium.

**Common infections:**
- Pneumonia
- Urinary tract infections
- Bloodstream infections
- Wound infections

**Effective antibiotics:**
- Ceftriaxone
- Meropenem
- Amikacin
- Piperacillin/tazobactam

**Resistance concerns:**
- Carbapenem-resistant Klebsiella (CRKP)
- ESBL-producing strains

**Match/Mismatch:** Critical to check susceptibility results."""
        
        else:
            return """**Most Common Organisms in Healthcare Infections:**

**Gram-Negative Bacteria:**
- **Escherichia coli (E. coli)**: UTI, bloodstream, GI infections
- **Klebsiella pneumoniae**: Pneumonia, UTI, bloodstream infections
- **Pseudomonas aeruginosa**: Hospital-acquired infections, wounds
- **Acinetobacter baumannii**: ICU infections, ventilator-associated
- **Proteus mirabilis**: UTI, wound infections

**Gram-Positive Bacteria:**
- **Staphylococcus aureus**: Skin, bloodstream, MRSA infections
- **Staphylococcus epidermidis**: Catheter-related infections
- **Streptococcus pneumoniae**: Pneumonia, meningitis
- **Enterococcus species**: UTI, bloodstream, endocarditis

**Fungi:**
- **Candida albicans**: Yeast infections, bloodstream
- **Aspergillus**: Invasive fungal infections

**Match/Mismatch:** Gram-negative organisms need different antibiotics than Gram-positive ones. Always check susceptibility results."""
    
    # Match/Mismatch queries
    elif any(word in prompt_lower for word in ['match', 'mismatch', 'susceptible', 'resistant']):
        return """**Understanding Drug-Bug Matches:**

**Match (Good):**
- Antibiotic is effective against the organism
- Organism is susceptible to the antibiotic
- Treatment likely to succeed

**Mismatch (Problem):**
- Antibiotic won't work against the organism
- Organism is resistant to the antibiotic
- Treatment likely to fail

**How to check:**
1. Look at susceptibility testing results
2. Check for "S" (Susceptible) vs "R" (Resistant)
3. Consider organism-antibiotic combinations
4. Review clinical guidelines

**Example:**
- E. coli + Ceftriaxone (S) = ✅ Match
- Pseudomonas + Vancomycin = ❌ Mismatch (Gram-negative vs Gram-positive)"""
    
    # General healthcare queries
    elif any(word in prompt_lower for word in ['help', 'how', 'what', 'when']):
        return """**I'm your Healthcare AI Assistant! 🤖**

I can help you with:
- **Antibiotic information** and uses
- **Organism details** and common infections
- **Drug-bug matching** and susceptibility
- **Treatment guidance** and best practices

**Try asking:**
- "Tell me about ceftriaxone"
- "What organisms does meropenem treat?"
- "How do I check if an antibiotic matches?"
- "What's the difference between match and mismatch?"

**Note:** I provide general information. Always consult healthcare professionals for specific medical advice."""
    
    # Default response
    else:
        return """I'm here to help with healthcare and antibiotic-related questions! 

**You can ask me about:**
- Specific antibiotics and their uses
- Organisms and infections
- Drug-bug matching
- Treatment guidelines
- Susceptibility testing

**Try asking something like:**
- "What is ceftriaxone used for?"
- "Tell me about E. coli infections"
- "How do I know if an antibiotic will work?"
- "What's the difference between match and mismatch?" """
