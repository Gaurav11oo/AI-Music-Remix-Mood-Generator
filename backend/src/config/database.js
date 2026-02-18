// const { Sequelize } = require('sequelize');
// require('dotenv').config();

// /*
// |--------------------------------------------------------------------------
// | Sequelize Instance
// |--------------------------------------------------------------------------
// */

// const sequelize = new Sequelize(
//   process.env.DB_NAME || 'music_remix_db',
//   process.env.DB_USER || 'postgres',
//   process.env.DB_PASSWORD,
//   {
//     host: process.env.DB_HOST || 'localhost',
//     port: process.env.DB_PORT || 5432,
//     dialect: 'postgres',

//     logging:
//       process.env.NODE_ENV === 'development'
//         ? (msg) => console.log('🟣 SQL:', msg)
//         : false,

//     pool: {
//       max: 10,
//       min: 0,
//       acquire: 30000,
//       idle: 10000,
//     },

//     define: {
//       timestamps: true,
//       underscored: true,
//       createdAt: 'created_at',
//       updatedAt: 'updated_at',
//     },

//     dialectOptions: {
//       statement_timeout: 30000,
//     },
//   }
// );

// /*
// |--------------------------------------------------------------------------
// | Test Database Connection
// |--------------------------------------------------------------------------
// */

// const testConnection = async () => {
//   try {
//     await sequelize.authenticate();
//     console.log('✅ Database connection established successfully');
//     return true;
//   } catch (error) {
//     console.error('❌ Unable to connect to database:', error.message);
//     return false;
//   }
// };

// /*
// |--------------------------------------------------------------------------
// | Sync Database Models (SAFE VERSION)
// |--------------------------------------------------------------------------
// |
// | IMPORTANT:
// | ❌ DO NOT USE alter:true (breaks Postgres constraints)
// | ❌ DO NOT AUTO MODIFY schema
// | ✅ Only create missing tables
// |
// */

// const syncDatabase = async (force = false) => {
//   try {
//     if (force) {
//       console.warn('⚠️ FORCE SYNC ENABLED — ALL TABLES WILL BE DROPPED');
//       await sequelize.sync({ force: true });
//     } else {
//       // ✅ SAFE SYNC (no ALTER)
//       await sequelize.sync();
//     }

//     console.log('✅ Database synchronized successfully');
//   } catch (error) {
//     console.error('❌ Database sync failed:', error.message);
//     throw error;
//   }
// };

// module.exports = {
//   sequelize,
//   testConnection,
//   syncDatabase,
// };


const { Sequelize } = require('sequelize');
require('dotenv').config();

/*
|--------------------------------------------------------------------------
| Database Configuration
|--------------------------------------------------------------------------
|
| Supports:
| ✅ Local PostgreSQL
| ✅ Neon / Railway / Supabase
| ✅ DATABASE_URL deployment
|
*/

const isProduction = process.env.NODE_ENV === 'production';

/*
|--------------------------------------------------------------------------
| Create Sequelize Instance
|--------------------------------------------------------------------------
*/

const sequelize = process.env.DATABASE_URL
  ? new Sequelize(process.env.DATABASE_URL, {
      dialect: 'postgres',

      logging: !isProduction
        ? (msg) => console.log('🟣 SQL:', msg)
        : false,

      dialectOptions: {
        ssl: {
          require: true,
          rejectUnauthorized: false, // required for Neon/Railway
        },
      },

      pool: {
        max: 10,
        min: 0,
        acquire: 30000,
        idle: 10000,
      },

      define: {
        timestamps: true,
        underscored: true,
        createdAt: 'created_at',
        updatedAt: 'updated_at',
      },
    })
  : new Sequelize(
      process.env.DB_NAME || 'music_remix_db',
      process.env.DB_USER || 'postgres',
      process.env.DB_PASSWORD,
      {
        host: process.env.DB_HOST || 'localhost',
        port: process.env.DB_PORT || 5432,
        dialect: 'postgres',

        logging: !isProduction
          ? (msg) => console.log('🟣 SQL:', msg)
          : false,

        pool: {
          max: 10,
          min: 0,
          acquire: 30000,
          idle: 10000,
        },

        define: {
          timestamps: true,
          underscored: true,
          createdAt: 'created_at',
          updatedAt: 'updated_at',
        },

        dialectOptions: {
          statement_timeout: 30000,
        },
      }
    );

/*
|--------------------------------------------------------------------------
| Test Database Connection
|--------------------------------------------------------------------------
*/

const testConnection = async () => {
  try {
    await sequelize.authenticate();
    console.log('✅ Database connection established successfully');
    return true;
  } catch (error) {
    console.error('❌ Unable to connect to database:', error.message);
    return false;
  }
};

/*
|--------------------------------------------------------------------------
| Sync Database Models (SAFE)
|--------------------------------------------------------------------------
*/

const syncDatabase = async (force = false) => {
  try {
    if (force) {
      console.warn('⚠️ FORCE SYNC ENABLED — ALL TABLES WILL BE DROPPED');
      await sequelize.sync({ force: true });
    } else {
      await sequelize.sync(); // safe sync
    }

    console.log('✅ Database synchronized successfully');
  } catch (error) {
    console.error('❌ Database sync failed:', error.message);
    throw error;
  }
};

module.exports = {
  sequelize,
  testConnection,
  syncDatabase,
};
