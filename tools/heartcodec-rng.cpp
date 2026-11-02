#include "heartcodec_rng.h"
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
int main(int argc,char ** argv) {
    heartcodec_rng_result_v1 r{};r.size=sizeof(r);
    try {
        if(argc!=9)throw std::runtime_error("Usage: heartcodec-rng normal|exponential SEED OFFSET COUNT DRAWS CAP DEVICE OUTPUT.f32");
        heartcodec_rng_config_v1 c{};c.size=sizeof(c);heartcodec_rng_config_init_v1(&c);
        const std::string kind=argv[1];if(kind!="normal"&&kind!="exponential")throw std::runtime_error("Unknown RNG distribution");
        c.distribution=kind=="normal"?0:1;c.seed=std::stoull(argv[2]);c.offset=std::stoull(argv[3]);c.count=std::stoll(argv[4]);
        c.draws=std::stoll(argv[5]);c.max_blocks=uint32_t(std::stoul(argv[6]));c.device=std::stoi(argv[7]);
        heartcodec_error_v1 e{};e.size=sizeof(e);
        if(heartcodec_rng_generate_v1(&c,&r,&e))throw std::runtime_error(e.message);
        std::ofstream out(argv[8],std::ios::binary);out.write(reinterpret_cast<const char *>(r.data),r.count*r.draws*4);if(!out)throw std::runtime_error("RNG output write failed");
        std::cout<<"{\"next_offset\":"<<r.next_offset<<",\"grid_blocks\":"<<r.grid_blocks<<",\"device_max_blocks\":"<<r.device_max_blocks<<"}\n";
        heartcodec_rng_result_free_v1(&r);return 0;
    }catch(const std::exception & e){heartcodec_rng_result_free_v1(&r);std::cerr<<e.what()<<'\n';return 1;}
}
