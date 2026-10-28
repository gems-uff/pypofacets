import ast
import re

def extract_functions(filepath):
    """Lê um arquivo Python e extrai um dicionário {nome_da_funcao: lista_de_linhas_limpas}."""
    with open(filepath, 'r', encoding='utf-8') as f:
        source = f.read()
    
    tree = ast.parse(source)
    lines = source.splitlines()
    functions = {}
    
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            # Pula a linha de definição 'def' e eventuais decoradores para focar só na lógica
            if node.body:
                start_line = node.body[0].lineno - 1
            else:
                start_line = node.lineno
            end_line = node.end_lineno
            
            func_lines = lines[start_line : end_line]
            cleaned_lines = preprocess_lines(func_lines)
            
            if cleaned_lines: # Só adiciona se o corpo da função não for vazio
                functions[node.name] = cleaned_lines
            
    return functions

def preprocess_lines(lines):
    """Remove espaços em branco nas pontas e ignora linhas vazias/comentários."""
    cleaned = []
    for line in lines:
        line = line.strip()
        if line and not line.startswith('#'):
            cleaned.append(line)
    return cleaned

def tokenize_line(line):
    """Transforma uma linha num set de tokens."""
    tokens = re.findall(r'[a-zA-Z_]\w*|\d+|[^\w\s]', line)
    return set(tokens)

def jaccard_similarity(set1, set2):
    """Calcula o Coeficiente de Jaccard entre dois conjuntos de tokens."""
    if not set1 and not set2: return 1.0
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union > 0 else 0.0

def line_similarity(line1, line2):
    """Calcula a similaridade entre duas linhas usando seus tokens."""
    return jaccard_similarity(tokenize_line(line1), tokenize_line(line2))

def function_similarity(lines_a, lines_b):
    """Calcula a similaridade entre duas funções usando alinhamento flexível de linhas."""
    if not lines_a and not lines_b: return 1.0
    if not lines_a or not lines_b: return 0.0
    
    score_a_to_b = sum(max(line_similarity(l_a, l_b) for l_b in lines_b) for l_a in lines_a)
    score_b_to_a = sum(max(line_similarity(l_b, l_a) for l_a in lines_a) for l_b in lines_b)
    
    return (score_a_to_b + score_b_to_a) / (len(lines_a) + len(lines_b))

def compare_directions(ref_funcs, cand_funcs, ref_name, cand_name):
    """Imprime a comparação de funções em um sentido específico."""
    print(f"\n{'='*100}")
    print(f"SENTIDO: {ref_name} -> {cand_name} (Referência: {ref_name})")
    print(f"{'='*100}")
    print(f"{'Função Origem ('+ref_name+')':<40} | {'Melhor Correspondência ('+cand_name+')':<35} | Similaridade")
    print("-" * 100)
    
    for name_ref, lines_ref in ref_funcs.items():
        best_match_name = None
        best_score = 0.0
        
        for name_cand, lines_cand in cand_funcs.items():
            score = function_similarity(lines_ref, lines_cand)
            if score > best_score:
                best_score = score
                best_match_name = name_cand
                
        match_display = best_match_name if best_match_name else "Nenhuma"
        print(f"{name_ref:<40} | {match_display:<35} | {best_score:.4f}")
    
def calcular_metricas_soft(funcs_gabarito, funcs_ia):
    print(f"\n{'-'*60}")
    print(f"MÉTRICAS SOFT BIDIRECIONAIS (Matematicamente Correto)")
    print(f"{'-'*60}")
    
    total_gabarito = len(funcs_gabarito)
    total_ia = len(funcs_ia)
    
    # Se uma das listas for vazia, evitamos erros
    if total_ia == 0 or total_gabarito == 0:
        print("Uma das listas de funções está vazia. Não é possível calcular.")
        return

    # 1. Ponto de vista do GABARITO (Foco na Revocação / Omissões)
    soft_tp_gabarito = 0.0
    for name_g, lines_g in funcs_gabarito.items():
        best_score = max((function_similarity(lines_g, lines_p) for lines_p in funcs_ia.values()), default=0.0)
        soft_tp_gabarito += best_score

    # 2. Ponto de vista da IA (Foco na Precisão / Excessos)
    soft_tp_ia = 0.0
    for name_p, lines_p in funcs_ia.items():
        best_score = max((function_similarity(lines_p, lines_g) for lines_g in funcs_gabarito.values()), default=0.0)
        soft_tp_ia += best_score

    # 3. Calculando os Erros (Garantidamente >= 0)
    soft_fn = total_gabarito - soft_tp_gabarito
    soft_fp = total_ia - soft_tp_ia
    
    # 4. Calculando as Métricas
    # Precisão: Do total que a IA fez, quanto era útil?
    soft_precision = soft_tp_ia / total_ia 
    
    # Recall: Do total que a IA devia fazer (Gabarito), quanto ela cobriu?
    soft_recall = soft_tp_gabarito / total_gabarito 
    
    # F1-Score: Média harmônica
    soft_f1 = 2 * (soft_precision * soft_recall) / (soft_precision + soft_recall) if (soft_precision + soft_recall) > 0 else 0.0
    
    print(f"Total Gabarito : {total_gabarito} funções originais")
    print(f"Total IA       : {total_ia} funções geradas\n")
    
    print(f"Soft FN (Omissões/Erros da IA) : {soft_fn:.2f} (Baseado no Gabarito)")
    print(f"Soft FP (Excesso/Alucinação)   : {soft_fp:.2f} (Baseado na IA)")
    print("-" * 60)
    print(f"Soft Precision : {soft_precision:.4f}  (O quanto o código gerado é preciso)")
    print(f"Soft Recall    : {soft_recall:.4f}  (O quanto o código original foi preservado)")
    print(f"Soft F1-Score  : {soft_f1:.4f}")

def main():
    file_gabarito = "modular_pypofacets.py"
    file_ia = "modular_pypofacets_gemini_v2.py"
    
    funcs_gabarito = extract_functions(file_gabarito)
    funcs_ia = extract_functions(file_ia)
    
    # Sentido 1: Gabarito como referência (Quais funções originais a IA cobriu?)
    compare_directions(funcs_gabarito, funcs_ia, "Gabarito", "IA")
    
    # Sentido 2: IA como referência (De onde vieram as funções que a IA criou?)
    compare_directions(funcs_ia, funcs_gabarito, "IA", "Gabarito")

    calcular_metricas_soft(funcs_gabarito, funcs_ia)

if __name__ == "__main__":
    main()