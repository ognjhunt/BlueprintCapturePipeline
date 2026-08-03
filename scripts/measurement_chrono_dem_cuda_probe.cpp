// Bounded exact-source Chrono::DEM CUDA development probe.
//
// This is a synthetic replay fixture. It does not establish characterized
// material behavior, Q-GRAN, physical success, production admission, or safety.

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "chrono/core/ChVector3.h"
#include "chrono_dem/physics/ChSystemDem.h"

using chrono::ChVector3f;
using chrono::dem::CHDEM_FRICTION_MODE;
using chrono::dem::CHDEM_OUTPUT_MODE;
using chrono::dem::CHDEM_ROLLING_MODE;
using chrono::dem::CHDEM_TIME_INTEGRATOR;
using chrono::dem::CHDEM_VERBOSITY;
using chrono::dem::ChSystemDem;

namespace {

constexpr const char* kSchema = "measurement_chrono_dem_cuda_probe_result.v1";
constexpr const char* kChronoVersion = "10.0.0";
constexpr const char* kSourceCommit = "9faf13dd8f1128dd75ed233a9627027b0422c3f7";
constexpr float kPi = 3.14159265358979323846f;
constexpr float kRadiusCm = 1.0f;
constexpr float kGravityCmS2 = -980.0f;
constexpr float kGroundZCm = -10.0f;
constexpr int kCountX = 3;
constexpr int kCountY = 3;
constexpr int kCountZ = 3;
constexpr int kParticleCount = kCountX * kCountY * kCountZ;

struct Args {
    float density_g_cm3 = 2.5f;
    float friction = 0.2f;
    float rolling_friction = 0.01f;
    float duration_s = 0.5f;
    float timestep_s = 5e-5f;
    float settle_speed_threshold_cm_s = 2.0f;
};

float parse_float(const char* raw, const char* name) {
    char* end = nullptr;
    const float value = std::strtof(raw, &end);
    if (end == raw || *end != '\0' || !std::isfinite(value)) {
        throw std::runtime_error(std::string("invalid argument: ") + name);
    }
    return value;
}

Args parse_args(int argc, char** argv) {
    Args args;
    for (int i = 1; i < argc; i += 2) {
        if (i + 1 >= argc) {
            throw std::runtime_error("missing argument value");
        }
        const std::string key = argv[i];
        const float value = parse_float(argv[i + 1], argv[i]);
        if (key == "--density-g-cm3") {
            args.density_g_cm3 = value;
        } else if (key == "--friction") {
            args.friction = value;
        } else if (key == "--rolling-friction") {
            args.rolling_friction = value;
        } else if (key == "--duration-s") {
            args.duration_s = value;
        } else if (key == "--timestep-s") {
            args.timestep_s = value;
        } else if (key == "--settle-speed-threshold-cm-s") {
            args.settle_speed_threshold_cm_s = value;
        } else {
            throw std::runtime_error(std::string("unknown argument: ") + key);
        }
    }
    if (!(args.density_g_cm3 >= 0.1f && args.density_g_cm3 <= 20.0f) ||
        !(args.friction >= 0.0f && args.friction <= 2.0f) ||
        !(args.rolling_friction >= 0.0f && args.rolling_friction <= 1.0f) ||
        !(args.duration_s >= 0.1f && args.duration_s <= 2.0f) ||
        !(args.timestep_s >= 1e-6f && args.timestep_s <= 1e-3f) ||
        !(args.settle_speed_threshold_cm_s > 0.0f && args.settle_speed_threshold_cm_s <= 100.0f)) {
        throw std::runtime_error("argument outside bounded development envelope");
    }
    const float steps = args.duration_s / args.timestep_s;
    if (steps < 100.0f || std::abs(steps - std::round(steps)) > 1e-3f) {
        throw std::runtime_error("duration and timestep are not integral");
    }
    return args;
}

void require_cuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ":" + cudaGetErrorName(status));
    }
}

std::string json_escape(const std::string& value) {
    std::string escaped;
    escaped.reserve(value.size());
    for (const char ch : value) {
        if (ch == '"' || ch == '\\') {
            escaped.push_back('\\');
            escaped.push_back(ch);
        } else if (static_cast<unsigned char>(ch) >= 0x20) {
            escaped.push_back(ch);
        }
    }
    return escaped;
}

struct Sample {
    float time_s;
    float centroid_x_cm;
    float centroid_y_cm;
    float centroid_z_cm;
    float horizontal_span_cm;
    float maximum_speed_cm_s;
    float settled_fraction;
    unsigned int contact_count;
    float kinetic_energy_native;
    float ground_reaction_force_n;
};

float horizontal_span(ChSystemDem& system) {
    float min_x = INFINITY;
    float max_x = -INFINITY;
    float min_y = INFINITY;
    float max_y = -INFINITY;
    for (int index = 0; index < kParticleCount; ++index) {
        const auto pos = system.GetParticlePosition(index);
        min_x = std::min(min_x, pos.x());
        max_x = std::max(max_x, pos.x());
        min_y = std::min(min_y, pos.y());
        max_y = std::max(max_y, pos.y());
    }
    return std::max(max_x - min_x, max_y - min_y);
}

Sample sample_system(ChSystemDem& system, size_t ground_id, const Args& args) {
    ChVector3f centroid(0.0f, 0.0f, 0.0f);
    float maximum_speed = 0.0f;
    int settled = 0;
    for (int index = 0; index < kParticleCount; ++index) {
        centroid += system.GetParticlePosition(index);
        const float speed = system.GetParticleVelocity(index).Length();
        maximum_speed = std::max(maximum_speed, speed);
        settled += speed < args.settle_speed_threshold_cm_s ? 1 : 0;
    }
    centroid /= static_cast<float>(kParticleCount);
    ChVector3f reaction(0.0f, 0.0f, 0.0f);
    if (!system.GetBCReactionForces(ground_id, reaction)) {
        throw std::runtime_error("ground reaction force unavailable");
    }
    return {
        system.GetSimTime(),
        centroid.x(),
        centroid.y(),
        centroid.z(),
        horizontal_span(system),
        maximum_speed,
        static_cast<float>(settled) / static_cast<float>(kParticleCount),
        system.GetNumContacts(),
        system.GetParticlesKineticEnergy(),
        std::abs(reaction.z()) * 1e-5f,
    };
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Args args = parse_args(argc, argv);
        int cuda_device_count = 0;
        require_cuda(cudaGetDeviceCount(&cuda_device_count), "cudaGetDeviceCount");
        if (cuda_device_count != 1) {
            throw std::runtime_error("exactly one CUDA device required");
        }
        require_cuda(cudaSetDevice(0), "cudaSetDevice");
        cudaDeviceProp device{};
        require_cuda(cudaGetDeviceProperties(&device, 0), "cudaGetDeviceProperties");

        ChSystemDem system(kRadiusCm, args.density_g_cm3, ChVector3f(30.0f, 30.0f, 30.0f));
        system.SetKn_SPH2SPH(5e7f);
        system.SetKn_SPH2WALL(5e7f);
        system.SetGn_SPH2SPH(2e4f);
        system.SetGn_SPH2WALL(2e4f);
        system.SetKt_SPH2SPH(2e7f);
        system.SetKt_SPH2WALL(2e7f);
        system.SetGt_SPH2SPH(1e4f);
        system.SetGt_SPH2WALL(1e4f);
        system.SetStaticFrictionCoeff_SPH2SPH(args.friction);
        system.SetStaticFrictionCoeff_SPH2WALL(args.friction);
        system.SetRollingCoeff_SPH2SPH(args.rolling_friction);
        system.SetRollingCoeff_SPH2WALL(args.rolling_friction);
        system.SetCohesionRatio(0.0f);
        system.SetAdhesionRatio_SPH2WALL(0.0f);
        system.SetGravitationalAcceleration(ChVector3f(0.0f, 0.0f, kGravityCmS2));
        system.SetParticleOutputMode(CHDEM_OUTPUT_MODE::NONE);
        system.SetBDFixed(true);
        system.SetFrictionMode(CHDEM_FRICTION_MODE::MULTI_STEP);
        system.SetRollingMode(CHDEM_ROLLING_MODE::SCHWARTZ);
        system.SetTimeIntegrator(CHDEM_TIME_INTEGRATOR::CENTERED_DIFFERENCE);
        system.SetFixedStepSize(args.timestep_s);
        system.SetVerbosity(CHDEM_VERBOSITY::QUIET);

        const float spacing = 2.1f * kRadiusCm;
        std::vector<ChVector3f> points;
        points.reserve(kParticleCount);
        for (int z = 0; z < kCountZ; ++z) {
            for (int y = 0; y < kCountY; ++y) {
                for (int x = 0; x < kCountX; ++x) {
                    const float stagger_x = z % 2 ? 0.48f * spacing : 0.0f;
                    const float stagger_y = z % 3 ? 0.31f * spacing : 0.0f;
                    points.emplace_back(
                        (x - 1) * spacing + stagger_x,
                        (y - 1) * spacing + stagger_y,
                        kGroundZCm + 1.5f * kRadiusCm + z * spacing);
                }
            }
        }
        system.SetParticles(points);
        const size_t ground_id =
            system.CreateBCPlane(ChVector3f(0.0f, 0.0f, kGroundZCm), ChVector3f(0.0f, 0.0f, 1.0f), true);
        system.Initialize();

        const float initial_span_cm = horizontal_span(system);
        const int sample_count = 20;
        const float sample_interval_s = args.duration_s / static_cast<float>(sample_count);
        std::vector<Sample> trace;
        trace.reserve(sample_count);
        for (int sample = 0; sample < sample_count; ++sample) {
            system.AdvanceSimulation(sample_interval_s);
            trace.push_back(sample_system(system, ground_id, args));
        }

        const float final_span_cm = horizontal_span(system);
        float maximum_reaction_force_n = 0.0f;
        unsigned int maximum_contact_count = 0;
        for (const auto& sample : trace) {
            maximum_reaction_force_n =
                std::max(maximum_reaction_force_n, sample.ground_reaction_force_n);
            maximum_contact_count = std::max(maximum_contact_count, sample.contact_count);
        }
        const float sphere_volume_cm3 =
            4.0f / 3.0f * kPi * std::pow(kRadiusCm, 3.0f);
        const float expected_weight_n =
            kParticleCount * sphere_volume_cm3 * args.density_g_cm3 * std::abs(kGravityCmS2) * 1e-5f;
        const float minimum_surface_z_cm = system.GetMinParticleZ() - kRadiusCm;
        const float penetration_m = std::max(0.0f, kGroundZCm - minimum_surface_z_cm) * 0.01f;

        std::cout << std::setprecision(9);
        std::cout << "{\"schema_version\":\"" << kSchema << "\",";
        std::cout << "\"status\":\"completed\",";
        std::cout << "\"chrono_version\":\"" << kChronoVersion << "\",";
        std::cout << "\"source_commit\":\"" << kSourceCommit << "\",";
        std::cout << "\"chrono_dem_module_used\":true,";
        std::cout << "\"cuda_device_count\":" << cuda_device_count << ',';
        std::cout << "\"cuda_device_name\":\"" << json_escape(device.name) << "\",";
        std::cout << "\"cuda_compute_capability\":\"" << device.major << '.' << device.minor
                  << "\",";
        std::cout << "\"particle_count\":" << kParticleCount << ',';
        std::cout << "\"density_g_cm3\":" << args.density_g_cm3 << ',';
        std::cout << "\"friction\":" << args.friction << ',';
        std::cout << "\"rolling_friction\":" << args.rolling_friction << ',';
        std::cout << "\"duration_s\":" << args.duration_s << ',';
        std::cout << "\"timestep_s\":" << args.timestep_s << ',';
        std::cout << "\"initial_horizontal_span_m\":" << initial_span_cm * 0.01f << ',';
        std::cout << "\"final_horizontal_span_m\":" << final_span_cm * 0.01f << ',';
        std::cout << "\"spread_ratio\":" << final_span_cm / initial_span_cm << ',';
        std::cout << "\"final_settled_fraction\":" << trace.back().settled_fraction << ',';
        std::cout << "\"final_maximum_speed_m_s\":" << trace.back().maximum_speed_cm_s * 0.01f
                  << ',';
        std::cout << "\"maximum_contact_count\":" << maximum_contact_count << ',';
        std::cout << "\"expected_static_weight_n\":" << expected_weight_n << ',';
        std::cout << "\"final_ground_reaction_force_n\":" << trace.back().ground_reaction_force_n
                  << ',';
        std::cout << "\"maximum_ground_reaction_force_n\":" << maximum_reaction_force_n << ',';
        std::cout << "\"penetration_m\":" << penetration_m << ',';
        std::cout << "\"trace\":[";
        for (size_t index = 0; index < trace.size(); ++index) {
            if (index) {
                std::cout << ',';
            }
            const auto& sample = trace[index];
            std::cout << "{\"time_s\":" << sample.time_s;
            std::cout << ",\"centroid_m\":[" << sample.centroid_x_cm * 0.01f << ','
                      << sample.centroid_y_cm * 0.01f << ',' << sample.centroid_z_cm * 0.01f << ']';
            std::cout << ",\"horizontal_span_m\":" << sample.horizontal_span_cm * 0.01f;
            std::cout << ",\"maximum_speed_m_s\":" << sample.maximum_speed_cm_s * 0.01f;
            std::cout << ",\"settled_fraction\":" << sample.settled_fraction;
            std::cout << ",\"contact_count\":" << sample.contact_count;
            std::cout << ",\"kinetic_energy_native\":" << sample.kinetic_energy_native;
            std::cout << ",\"ground_reaction_force_n\":" << sample.ground_reaction_force_n << '}';
        }
        std::cout << "]}" << std::endl;
        return 0;
    } catch (const std::exception& exc) {
        std::cerr << "measurement_chrono_dem_cuda_probe_failed:" << exc.what() << std::endl;
        return 2;
    }
}
