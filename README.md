# NoiseMap-RT : Cartographie du Bruit des Drones par Ray Tracing

Ce projet implémente un système de cartographie acoustique spécialisé pour les drones, utilisant le ray tracing pour calculer la propagation du bruit autour d'une source ponctuelle sur un maillage 3D. Il utilise OptiX (NVIDIA) pour l'accélération GPU du ray tracing et des modèles acoustiques avancés basés sur les normes ISO 9613-1 et des études spécifiques aux drones pour la simulation réaliste de la propagation du son.

## Architecture Générale

Le code est organisé autour de plusieurs "composants" principaux :

- **Point d'entrée** : `main.cpp` orchestre l'ensemble du processus
- **Gestion de scène** : `scene.h/cpp` gère le chargement du maillage, le ray tracing et l'export
- **Ray tracing GPU** : `RayTracer/ray_tracer.h/cpp` et `RayTracer/optix_ray.h/cu` pour le calcul des visibilités
- **Modèle acoustique** : `RayTracer/acoustic_model.h/cpp` pour les calculs de niveaux sonores avec modèles spécifiques aux drones

### Flux de données
1. Chargement du maillage PLY
2. Calcul des rayons depuis la source vers chaque face du maillage
3. Détermination des visibilités (ray tracing)
4. Calcul des niveaux sonores (SPL) avec modèles acoustiques incluant les effets de sol, absorption atmosphérique et directivité
5. Colorisation du maillage selon les niveaux de bruit
6. Export du maillage coloré en PLY

## Détail des Fichiers C++

### main.cpp
**Rôle** : Point d'entrée principal du programme. Gère le parsing des arguments, la configuration du logger et l'orchestration des calculs.

**Fonctions principales** :
- `setupLogger()` : Configure spdlog avec sortie console et fichier
- `printUsage(const char* progname)` : Affiche l'aide d'utilisation
- `parseArguments(int argc, char* argv[], ...)` : Parse les arguments de ligne de commande
- `main(int argc, char* argv[])` : Fonction principale orchestrant :
  - Chargement de la scène
  - Ray tracing
  - Calcul acoustique
  - Colorisation
  - Export PLY

**Utilisation** : Lance le programme avec les paramètres de source et environnement acoustique.

### scene.h / scene.cpp
**Rôle** : Classe centrale `Scene` gérant le maillage 3D, les calculs de visibilité et l'export.

**Classe Scene** :
- **Constructeurs** :
  - `Scene(const std::string& ply_file)` : Charge un maillage depuis un fichier PLY *(déclaré mais non défini dans le code fourni)*
  - `Scene(SurfaceMesh& mesh)` : Construit une scène depuis un maillage CGAL existant

**Méthodes publiques** :
- `traceRays(const Point& point)` : Calcule les distances de visibilité pour un point source
- `traceRays(const std::vector<Point>& points)` : Version batch pour plusieurs points
- `computeNoiseMap(const std::vector<float>& distances, const AcousticModel& model)` : Calcule les SPL dBA
- `addDistances(const std::vector<float>& distances)` : Ajoute les distances comme propriété du maillage
- `addSPL(const std::vector<double>& spl_values)` : Ajoute les SPL comme propriété
- `addColor(const std::vector<float>& values)` : Colorise selon les distances
- `addNoiseMapColor(const std::vector<double>& spl_values)` : Colorise selon les niveaux de bruit
- `writeMeshToPLY(const std::string& filename)` : Exporte le maillage avec propriétés

**Fonctions utilitaires** :
- `getColor(double value, double minValue, double maxValue)` : Échelle de couleur plasma
- `getNoiseColor(double spl_dBA, double minSPL, double maxSPL)` : Échelle de couleur pour le bruit (ISO)

**Utilisation** : Interface principale pour manipuler le maillage et effectuer tous les calculs.

### RayTracer/ray_tracer.h / ray_tracer.cpp
**Rôle** : Classe `RayTracer` gérant le ray tracing accéléré GPU via OptiX pour déterminer les visibilités entre la source et les faces du maillage.

**Classe RayTracer** :
- **Constructeur** : `RayTracer(SurfaceMesh& mesh)` : Initialise OptiX, construit l'Acceleration Structure Géométrique (GAS), compile le module PTX, crée les program groups et le pipeline
- **Méthodes principales** :
  - `traceRay(const Point& p)` : Calcule les distances de visibilité pour un point source unique
  - `traceRay(const std::vector<Point>& points)` : Version batch pour plusieurs points sources
  - `cleanup()` : Libère les ressources GPU

**Méthodes privées** :
- `computeRaysAndHits(...)` : Implémente le ray tracing OptiX pour déterminer les occultations :
  - Construction des rayons depuis la source vers le centroïde de chaque face
  - Upload des rayons vers GPU
  - Lancement du pipeline OptiX avec shader raygen, miss et closest-hit
  - Récupération des résultats de visibilité
  - Retour des distances (positive si visible, -1 si occultée)

**Fonctionnalités OptiX** :
- Initialisation du contexte CUDA et OptiX
- Construction de GAS triangulée à partir du maillage CGAL
- Compilation du module PTX embarqué (optix_ray.ptx)
- Création des program groups (raygen, miss, hitgroup)
- Configuration du pipeline avec paramètres de lancement
- Shader Binding Table (SBT) pour l'exécution des shaders

**Utilisation** : Effectue le calcul des visibilités entre la source et chaque face du maillage avec accélération GPU.

### RayTracer/acoustic_model.h / acoustic_model.cpp
**Rôle** : Classe `AcousticModel` implémentant les modèles de propagation acoustique avancés spécialisés pour les drones.

**Structures et Classes** :
- `DroneDatabase` : Base de données singleton contenant les modèles d'émission pour différents drones
- `DroneEmissionModel` : Structure décrivant un modèle de drone avec niveaux de référence, pentes RPM et paramètres
- `AcousticParams` : Paramètres acoustiques (puissance source, hauteurs, conditions atmosphériques, type de sol, modèle de drone)
- `GroundType` : Énumération des types de sol (ASPHALT, COMPACT_SOIL, GRASS)
- `FlightManoeuvre` : Énumération des manœuvres de vol (HOVER, TAKEOFF, LANDING, CRUISE)

**Classe AcousticModel** :
- **Constructeur** : `AcousticModel(const AcousticParams& params)` : Initialise avec les paramètres et la résistivité du sol
- **Méthodes principales** :
  - `computeSPLSpectrum(double distance_m, bool visible, double out_bands[NUM_BANDS])` : Calcule le spectre SPL par bande de tiers d'octave
  - `computeSPL(double distance_m, bool visible)` : Calcule le SPL global A-pondéré [dBA]

**Modèles de propagation acoustique implémentés** :
- `geometricalSpreading(double distance_m)` : Atténuation géométrique (1/r²)
- `atmosphericAbsorption(double freq_Hz, double distance_m)` : Absorption atmosphérique (ISO 9613-1)
- `directivityCorrection(double theta_deg, double freq_Hz)` : Directivité verticale de la source avec filtre high-shelving
- `groundEffect(double freq_Hz, double distance_m)` : Réflexion au sol (Delany-Bazley avec interférences)
- `rpmEqualization(double freq_Hz, double rpm_actual, const DroneEmissionModel& model)` : Correction RPM selon l'équation 2

**Modèles de drones supportés** :
- **Mavic 2 Pro (M2)** : Quadricoptère léger, RPM référence 6540 r/min
- **Inspire 2 (I2)** : Quadricoptère professionnel, RPM référence 4560 r/min
- **F-450 (F-4)** : Quadricoptère agricole, RPM référence 6420 r/min
- **S-900 (S-9)** : Hexacoptère lourd, RPM référence 6900 r/min

**Paramètres de manœuvres** : Corrections SPL pour différentes phases de vol (survol stationnaire, décollage, atterrissage, croisière)

**Constantes** :
- `THIRD_OCTAVE_FREQS[29]` : Fréquences des bandes de tiers d'octave (25 Hz - 16 kHz)
- `A_WEIGHTING[29]` : Corrections A-pondération pour chaque bande

**Utilisation** : Calcule les niveaux sonores réels en tenant compte de tous les phénomènes de propagation.

### RayTracer/optix_ray.h / optix_ray.cu
**Rôle** : Code CUDA/OptiX pour les shaders de ray tracing.

**Structures** :
- `OptixRay` : Structure rayon (origine, direction, distance max)
- `RayGenLaunchParams` : Paramètres de lancement pour le kernel

**Shaders OptiX** :
- `__raygen__rg()` : Shader de génération de rayons - lance un rayon par face
- `__closesthit__ch()` : Shader de collision - marque quand un rayon touche une géométrie
- `__miss__ms()` : Shader de miss - appelé quand un rayon ne touche rien

**Utilisation** : Exécuté sur GPU pour déterminer si chaque face est visible depuis la source.

## Dépendances

- **CGAL** : Géométrie computationnelle pour les maillages
- **spdlog** : Logging performant
- **CUDA Toolkit** : Programmation GPU NVIDIA
- **OptiX 9.0** : API de ray tracing accéléré
- **CMake 3.14+** : Système de build

## Compilation

```bash
mkdir build && cd build
cmake ..
make
```

## Utilisation

```bash
./NoiseMap <fichier.ply> <x> <y> <z> [options]

Options :
  --ground asphalt|soil|grass    : Type de sol
  --temp <celsius>              : Température (°C)
  --humidity <percent>          : Humidité relative (%)
  --source-height <meters>      : Hauteur source (m)
  --receiver-height <meters>    : Hauteur récepteur (m)
```

## Sortie

Le programme génère un fichier `<input>_noisemap.ply` avec :
- Couleurs par face selon les niveaux de bruit (échelle ISO)
- Propriétés par face : distance, spl_dBA
- Géométrie identique à l'entrée

## Modèles Acoustiques

- **Propagation géométrique** : Atténuation 1/r²
- **Absorption atmosphérique** : ISO 9613-1 (température, humidité, pression)
- **Directivité verticale** : Ajustée pour sources aériennes (drones)
- **Réflexion au sol** : Modèle Delany-Bazley avec interférence

## Limitations

- Source ponctuelle uniquement
- Maillages triangulaires
- Calcul par face (pas par vertex)
- GPU NVIDIA requis (OptiX)



Modification appliquée : Les surfaces occultées ne reçoivent aucun son (SPL = −∞) et restent en gris foncé. Le calcul acoustique (`computeNoiseMap`) ignore les faces occultées (distance < 0). La colorisation (`addNoiseMapColor`) fixe la couleur de face des faces occultées en gris foncé et réinitialise également en gris foncé les sommets appartenant à une face occultée, évitant tout saignement de couleur par interpolation lors du rendu.
