import { ImageResponse } from 'next/og'
 
// Image metadata
export const size = {
  width: 32,
  height: 32,
}
export const contentType = 'image/png'
 
// Image generation
export default function Icon() {
  return new ImageResponse(
    (
      // ImageResponse JSX element
      <div
        style={{
          fontSize: 24,
          background: 'linear-gradient(135deg, #6366f1 0%, #ec4899 100%)',
          width: '100%',
          height: '100%',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          borderRadius: '20%',
          boxShadow: '0 0 10px rgba(99, 102, 241, 0.5)',
        }}
      >
        <span style={{ color: 'white', marginTop: -2 }}>⭐</span>
      </div>
    ),
    // ImageResponse options
    {
      ...size,
    }
  )
}
