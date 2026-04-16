/*
 * Created on Fri Oct 03 2025
 * Copyright (c) 2025 HENRY Antoine
 * Licensed under the Apache License, Version 2.0
 * Additional Restriction: This code may not be used for commercial purposes.
 */

#pragma once

#include <array>
#include <string>
#include <unordered_map>
#include <vector>

// ─── Constantes spectrales ────────────────────────────────────────────────────

static constexpr int NUM_BANDS = 24;

/// Fréquences centrales des bandes de tiers d'octave (100 Hz → 16 kHz)
/// Conformément au Tableau 3 du papier (Heutschi et al., 2020)
static constexpr double THIRD_OCTAVE_FREQS[NUM_BANDS] = {
    100, 125, 160, 200, 250, 315, 400, 500, 630, 800,
    1000, 1250, 1600, 2000, 2500, 3150, 4000, 5000,
    6300, 8000, 10000, 12500, 16000, 20000
};

/// Pondération A [dB] pour chaque bande de tiers d'octave
static constexpr double A_WEIGHTING[NUM_BANDS] = {
    -19.1, -16.1, -13.4, -10.9, -8.6, -6.6, -4.8, -3.2, -1.9, -0.8,
     0.0,   0.6,   1.0,   1.2,   1.3,  1.2,  1.0,  0.5,
    -0.1,  -1.1,  -2.5,  -4.3,  -6.6, -9.3
};

// ─── Types énumérés ───────────────────────────────────────────────────────────

/// Types de sol supportés (résistivités tirées de Delany & Bazley 1970)
enum class GroundType {
    ASPHALT,        ///< Asphalte : ~20 000 kPa·s/m²
    COMPACT_SOIL,   ///< Sol compact : ~5 000 kPa·s/m²
    GRASS           ///< Herbe : ~300 kPa·s/m²
};

/// Manœuvres de vol (Section 3 du papier)
enum class FlightManoeuvre {
    HOVER_10M,      ///< Vol stationnaire à 10 m
    HOVER_20M,      ///< Vol stationnaire à 20 m
    HOVER_50M,      ///< Vol stationnaire à 50 m
    CLIMB,          ///< Montée (20–50 m)
    SINK,           ///< Descente (50–10 m)
    FORWARD_DOWNWIND, ///< Vol en avant, vent dans le dos
    FORWARD_UPWIND    ///< Vol en avant, vent de face
};

// ─── Modèle d'émission par type de drone ──────────────────────────────────────

/**
 * @brief Paramètres d'émission pour un type de drone spécifique.
 *
 * Basé sur le Tableau 2 (vitesses de rotation de référence),
 * le Tableau 3 (pentes de l'égaliseur S [dB/r/min]) et
 * la Figure 6 (spectres de base) du papier Heutschi et al. (2020).
 */
struct DroneEmissionModel {
    std::string name;           ///< Identifiant du drone (ex. "M2", "I2", "F-4", "S-9")
    int         num_rotors;     ///< Nombre de rotors
    double      weight_g;       ///< Masse [g]
    double      rpm_ref;        ///< Vitesse de rotation de référence R_ref [r/min]

    /**
     * @brief Niveaux de pression acoustique de référence par bande [dB re 20 µPa @ 1.5 m].
     *
     * Correspond aux spectres de la Figure 6 à R_ref, distance 1.5 m,
     * angle d'élévation −30° par rapport au plan des rotors.
     */
    std::array<double, NUM_BANDS> Lp_ref;

    /**
     * @brief Pentes de l'égaliseur S(i) [dB / (r/min)] par bande de tiers d'octave.
     *
     * Selon l'équation (2) : E(i) = S(i) * (R − R_ref)
     * Valeurs tirées du Tableau 3.
     */
    std::array<double, NUM_BANDS> equalizer_slope;
};

// ─── Paramètres de manœuvre ───────────────────────────────────────────────────

/**
 * @brief Paramètres de variation du rpm liés à la manœuvre et au vent.
 *
 * Modèle linéaire (Eq. 4) : σn = a + b * |v_wind|
 * Valeurs tirées du Tableau 5.
 */
struct ManoeuvreParams {
    double rpm_average;   ///< Vitesse de rotation moyenne pour cette manœuvre [r/min]
                          ///< Tirée du Tableau 4 (vitesses par manœuvre)
    double sigma_a;       ///< Ordonnée à l'origine du modèle σn (adimensionnel)
    double sigma_b;       ///< Pente du modèle σn [(s/m)]
};

// ─── Paramètres globaux du modèle ─────────────────────────────────────────────

/**
 * @brief Paramètres complets pour le calcul acoustique d'un scénario de drone.
 */
struct AcousticParams {
    // -- Échelle & référentiel --
    double unit_scale      = 100.0;  ///< 1 unité mesh = unit_scale mètres (défaut 100)
    int    reflection_order = 1;     ///< Ordre de réflexion : 0 = direct seul, 1 = +un trajet réfléchi séparé

    // -- Source --
    double source_height   = -1.0;   ///< Hauteur source [m] (-1 = auto depuis Z × scale)
    double receiver_height =  0.0;   ///< Hauteur récepteur par défaut [m]

    /// Niveaux de puissance acoustique par bande [dB re 1 pW]
    /// (utilisé si aucun DroneEmissionModel n'est fourni)
    std::array<double, NUM_BANDS> source_Lw{};

    // -- Atmosphère (ISO 9613-1) --
    double temperature_C  = 15.0;   ///< Température [°C]
    double humidity_pct   = 70.0;   ///< Humidité relative [%]
    double pressure_kPa   = 101.325; ///< Pression atmosphérique [kPa]

    // -- Sol --
    GroundType ground_type = GroundType::GRASS;

    // -- Drone (optionnel) --
    const DroneEmissionModel* drone_model = nullptr; ///< Pointeur vers le modèle de drone

    // -- Conditions de vol --
    FlightManoeuvre manoeuvre = FlightManoeuvre::HOVER_10M;
    double          rpm_actual = 0.0;    ///< RPM réel de vol (0 = utiliser la moyenne de manœuvre)
    double          wind_speed_2_5m = 2.0; ///< Vitesse du vent à 2.5 m [m/s]

    // -- Distance de référence de l'enregistrement lab --
    double d_ref = 1.5;   ///< Distance d'enregistrement en chambre anéchoïque [m]
};

// ─── Base de données des modèles de drones ────────────────────────────────────

/**
 * @brief Fournit les modèles d'émission prédéfinis pour les drones du papier.
 *
 * Drones supportés : "M2" (Mavic 2 Pro), "I2" (Inspire 2),
 *                    "F-4" (F-450), "S-9" (S-900).
 */
class DroneDatabase {
public:
    /// Retourne l'instance singleton de la base de données
    static const DroneDatabase& instance();

    /**
     * @brief Cherche un modèle de drone par son identifiant.
     * @param name  Identifiant du drone ("M2", "I2", "F-4", "S-9")
     * @return Pointeur vers le modèle, nullptr si non trouvé
     */
    const DroneEmissionModel* find(const std::string& name) const;

    /// Retourne tous les modèles disponibles
    const std::vector<DroneEmissionModel>& all() const { return models_; }

private:
    DroneDatabase();
    std::vector<DroneEmissionModel>                    models_;
    std::unordered_map<std::string, size_t>            index_;
};

// ─── Modèle acoustique principal ──────────────────────────────────────────────

/**
 * @brief Calcul du niveau de pression acoustique (SPL) en champ libre.
 *
 * Intègre les effets de :
 *  - émission du drone (modèle rpm + égalisation spectrale)
 *  - directivité verticale générique (filtre high-shelving, Eq. 1)
 *  - divergence géométrique (20·log₁₀(r))
 *  - absorption atmosphérique (ISO 9613-1)
 *  - effet de sol (Delany-Bazley + interférences)
 *  - correction de manœuvre et RPM (Eq. 2, Tableau 3)
 *
 * Référence : Heutschi et al., Acta Acustica 2020, 4, 24.
 */
class AcousticModel {
public:
    explicit AcousticModel(const AcousticParams& params);

    /// Accès en lecture aux paramètres (utilisé par Scene pour la géométrie)
    const AcousticParams& params() const { return params_; }

    // ── Calcul principal ──────────────────────────────────────────────────────

    /**
     * @brief Calcule le SPL pondéré A global [dB(A)].
     * @param distance_m   Distance source–récepteur [m]
     * @param visible      Ligne de vue directe disponible
     * @param direct_only  Si true, ne pas inclure l'effet de sol (groundEffect)
     *                     — utilisé quand le rayon réfléchi est traité séparément
     */
    double computeSPL(double distance_m, bool visible,
                      bool direct_only = false) const;

    /**
     * @brief Calcule le spectre SPL par bande de tiers d'octave [dB].
     * @param distance_m   Distance source–récepteur [m]
     * @param visible      Ligne de vue directe disponible
     * @param out_bands    Tableau de sortie (NUM_BANDS valeurs)
     * @param direct_only  Si true, ne pas inclure l'effet de sol (groundEffect)
     */
    void computeSPLSpectrum(double distance_m, bool visible,
                            double out_bands[NUM_BANDS],
                            bool direct_only = false) const;

    // ── Sous-modèles exposés publiquement ────────────────────────────────────

    /// Atténuation par divergence géométrique [dB]
    static double geometricalSpreading(double distance_m);

    /**
     * @brief Correction de directivité verticale [dB].
     *
     * Modèle polynomial (Eq. 1) couplé à un filtre high-shelving
     * de coin à 500 Hz (Q=0.5), valable pour les multi-rotors.
     *
     * @param theta_deg  Angle d'élévation depuis le plan horizontal [−90…+90°]
     *                   Positif = récepteur au-dessus de la source
     * @param freq_Hz    Fréquence centrale de la bande [Hz]
     */
    static double directivityCorrection(double theta_deg, double freq_Hz);

    /**
     * @brief Calcule l'angle d'élévation vu depuis la source.
     * @param horiz_m  Distance horizontale [m]
     * @param dz_m     Différence de hauteur source − récepteur [m]
     */
    static double elevationAngle(double horiz_m, double dz_m);

    /// Absorption atmosphérique cumulée [dB] sur la distance donnée
    double atmosphericAbsorption(double freq_Hz, double distance_m) const;

    /// Coefficient d'absorption atmosphérique [dB/m]
    double absorptionCoefficient(double freq_Hz) const;

    /// Effet de sol par interférences directe + réfléchie [dB]
    double groundEffect(double freq_Hz, double distance_m) const;

    /**
     * @brief Calcule le SPL pondéré A pour un trajet réfléchi seul [dB(A)].
     *
     * Utilisé pour les faces occultées du trajet direct mais qui reçoivent
     * le son via la réflexion au sol (méthode source-image).
     * Applique : Lw − divergence(d_refl) − absorption_atm(d_refl) + R_p(f,ψ)
     *
     * @warning Ne PAS additionner énergétiquement avec computeSPL ou
     * computeSPLSpectrum sur le même récepteur : ces dernières incluent déjà
     * la contribution réfléchie via groundEffect (interférence cohérente
     * direct+réfléchi). Cette fonction est à utiliser exclusivement pour les
     * récepteurs dont le trajet direct est occulté.
     *
     * @param d_horiz  Distance horizontale source–récepteur [m]
     * @param hs       Hauteur de la source au-dessus du sol [m]
     * @param hr       Hauteur du récepteur au-dessus du sol [m]
     */
    double computeReflectedSPL(double d_horiz, double hs, double hr) const;

    // ── Modèles spécifiques aux drones (Section 2.3 & 3) ─────────────────────

    /**
     * @brief Calcule l'égalisation spectrale due au changement de RPM.
     *
     * Applique l'équation (2) du papier :
     *   E(i) = S(i) * (R_actual − R_ref)
     *
     * @param freq_Hz    Fréquence centrale de la bande [Hz]
     * @param rpm_actual RPM réel de vol
     * @param model      Modèle d'émission du drone
     * @return           Correction à appliquer [dB]
     */
    static double rpmEqualization(double freq_Hz, double rpm_actual,
                                  const DroneEmissionModel& model);

    /**
     * @brief Retourne les paramètres de manœuvre (RPM moyen + coefficients σn).
     *
     * Source : Tableaux 4 et 5 du papier.
     *
     * @param manoeuvre  Manœuvre de vol
     * @param model      Modèle de drone (pour les RPM moyens)
     * @return           Paramètres de manœuvre
     */
    static ManoeuvreParams getManoeuvreParams(FlightManoeuvre manoeuvre,
                                              const DroneEmissionModel& model);

    /**
     * @brief Calcule la déviation standard normalisée du RPM.
     *
     * Équation (4) : σn = a + b * |v_wind|
     *
     * @param manoeuvre        Manœuvre de vol
     * @param wind_speed_ms    Vitesse du vent à 2.5 m [m/s]
     * @param model            Modèle de drone
     * @return                 σn (adimensionnel)
     */
    static double rpmNormalisedStdDev(FlightManoeuvre manoeuvre,
                                      double wind_speed_ms,
                                      const DroneEmissionModel& model);
    static void computeDroneLw(const DroneEmissionModel& model,
                               double rpm_actual,
                               double out_Lw[NUM_BANDS],
                               double d_ref = 1.5);

private:
    AcousticParams params_;
    double         ground_resistivity_;  ///< σ [Pa·s/m²]

    static double getGroundResistivity(GroundType type);
};
