#include <mpi.h>
#include <string>
#include <map>

class ProfilerSection {
private:
    std::string name;
    double start_time;
    static std::map<std::string, double> timings;

public:
    ProfilerSection(const std::string& section_name) 
        : name(section_name) {
        MPI_Barrier(MPI_COMM_WORLD);
        start_time = MPI_Wtime();
    }
    
    ~ProfilerSection() {
        double elapsed = MPI_Wtime() - start_time;
        timings[name] += elapsed;
    }
    
    static void print_summary();
}; 