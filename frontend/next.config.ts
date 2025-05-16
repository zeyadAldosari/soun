/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  webpack: (config: any) => {
    // This is necessary for webworkers to function correctly
    config.resolve.fallback = {
      fs: false,
      path: false,
      crypto: false,
    };
    
    // Increase the maximum asset size to accommodate large DICOM files
    config.performance = {
      ...config.performance,
      maxAssetSize: 10 * 1024 * 1024, // 10MB
      maxEntrypointSize: 10 * 1024 * 1024, // 10MB
    };
    
    return config;
  },
}

module.exports = nextConfig