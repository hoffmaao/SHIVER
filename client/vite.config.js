import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'
import Sitemap from 'vite-plugin-sitemap'

const shiverRoutes = [
  '/SHIVER/',
  '/SHIVER/map',
  '/SHIVER/cube',
  '/SHIVER/documentation/overview',
  '/SHIVER/documentation/whatiszarr',
  '/SHIVER/documentation/contributingdata',
  '/SHIVER/documentation/unifieddata',
  '/SHIVER/documentation/datacomparison',
  '/SHIVER/documentation/timeseriesexplore',
  '/SHIVER/documentation/netcdfextract',
  '/SHIVER/documentation/cloud',
  '/SHIVER/documentation/citation',
  '/SHIVER/documentation/projects/overview',
  '/SHIVER/documentation/projects/people'
]

export default defineConfig({
  base: '/SHIVER/', 
  plugins: [
    vue(),
    Sitemap({
      // 2. Use the bare domain (the subfolder is handled by your routes array)
      hostname: 'https://shiver-ice-velocity.github.io', 
      
      // 3. Tell the plugin to ignore its incomplete auto-scan root
      exclude: ['/'],
      
      // 4. Force feed the explicit routes so timing issues don't matter
      dynamicRoutes: shiverRoutes
    })
  ],
  worker: {
    // The timeseries extraction worker is bundled separately, and Vite runs
    // the main plugin list over that bundle too. The sitemap plugin writes
    // into dist/ from closeBundle, which has not been created yet at that
    // point, so the worker bundle gets an empty plugin list of its own.
    format: 'es',
    plugins: () => []
  },
  ssr: {
    noExternal: ['file-saver', '@unhead/vue', 'unhead']
  }
})