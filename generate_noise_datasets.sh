#!/bin/bash

# Script per generare dataset con rumore a intensità crescenti
# Genera 3 livelli di intensità per ogni tipo di rumore

echo "=== Generazione Dataset con Rumore ==="
echo "Inizio: $(date)"
echo

# Verifica che lo script Python esista
if [ ! -f "add_noise.py" ]; then
    echo "Errore: Script add_noise.py non trovato nella directory corrente"
    exit 1
fi

# Verifica che la directory dataset_rgb esista
if [ ! -d "/home/brusc/Projects/random_forest/dataset_rgb" ]; then
    echo "Errore: Directory dataset_rgb non trovata"
    exit 1
fi

# Contatore per il progresso
total_tasks=15
current_task=0

# Funzione per eseguire il comando e mostrare il progresso
run_noise_command() {
    local noise_type=$1
    local intensity=$2
    
    current_task=$((current_task + 1))
    echo "[$current_task/$total_tasks] Generando $noise_type con intensità $intensity..."
    
    python add_noise.py --noise-type "$noise_type" --intensity "$intensity"
    
    if [ $? -eq 0 ]; then
        echo "✓ Completato: dataset_rgb_${noise_type}_${intensity}"
    else
        echo "✗ Errore durante la generazione di $noise_type intensità $intensity"
        return 1
    fi
    echo
}

# Rumore Gaussiano - intensità: 10, 30, 50
echo "=== Rumore Gaussiano ==="
run_noise_command "gaussian" 10
run_noise_command "gaussian" 30
run_noise_command "gaussian" 50

# Rumore Sale e Pepe - intensità: 5, 15, 25
echo "=== Rumore Sale e Pepe ==="
run_noise_command "salt_and_pepper" 5
run_noise_command "salt_and_pepper" 15
run_noise_command "salt_and_pepper" 25

# Rumore Speckle - intensità: 15, 35, 55
echo "=== Rumore Speckle ==="
run_noise_command "speckle" 15
run_noise_command "speckle" 35
run_noise_command "speckle" 55

# Rumore Poisson - intensità: 20, 40, 60
echo "=== Rumore Poisson ==="
run_noise_command "poisson" 20
run_noise_command "poisson" 40
run_noise_command "poisson" 60

# Rumore Uniforme - intensità: 10, 25, 40
echo "=== Rumore Uniforme ==="
run_noise_command "uniform" 10
run_noise_command "uniform" 25
run_noise_command "uniform" 40

echo "=== Riepilogo ==="
echo "Completato: $(date)"
echo

# Mostra le cartelle generate
echo "Cartelle generate:"
ls -la | grep "dataset_rgb_" | awk '{print "- " $9}'

echo
echo "Totale cartelle generate: $(ls -la | grep "dataset_rgb_" | wc -l)"

# Calcola lo spazio utilizzato
echo "Spazio utilizzato dalle nuove cartelle:"
du -sh dataset_rgb_* 2>/dev/null | sort -h

echo
echo "=== Processo Completato ==="