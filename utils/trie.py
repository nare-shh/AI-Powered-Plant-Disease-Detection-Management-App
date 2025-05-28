class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end = False
        self.disease_info = None

class DiseaseSymptomTrie:
    def __init__(self):
        self.root = TrieNode()
    
    def insert(self, symptoms: list, disease_info: dict):
        node = self.root
        for symptom in symptoms:
            if symptom not in node.children:
                node.children[symptom] = TrieNode()
            node = node.children[symptom]
        node.is_end = True
        node.disease_info = disease_info