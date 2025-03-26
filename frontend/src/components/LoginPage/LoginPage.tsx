import React from 'react'
import './LoginPage.css'

function LoginPage() {
  const handleGoogleLogin = () => {
    const clientId = 'YOUR_GOOGLE_CLIENT_ID';
    const redirectUri = encodeURIComponent('http://localhost:5173/auth/google/callback');
    const scope = encodeURIComponent('email profile');
    const url = `https://accounts.google.com/o/oauth2/v2/auth?client_id=${clientId}&redirect_uri=${redirectUri}&response_type=code&scope=${scope}`;
    window.location.href = url;
  };

  const handleAppleLogin = () => {
    const clientId = 'YOUR_APPLE_CLIENT_ID';
    const redirectUri = encodeURIComponent('http://localhost:5173/auth/apple/callback');
    const url = `https://appleid.apple.com/auth/authorize?client_id=${clientId}&redirect_uri=${redirectUri}&response_type=code&scope=email name&response_mode=form_post`;
    window.location.href = url;
  };

  const handleMetaLogin = () => {
    const appId = 'YOUR_META_APP_ID';
    const redirectUri = encodeURIComponent('http://localhost:5173/auth/meta/callback');
    const url = `https://www.facebook.com/v12.0/dialog/oauth?client_id=${appId}&redirect_uri=${redirectUri}&scope=email,public_profile`;
    window.location.href = url;
  };

  const handleEmailLogin = () => {
    window.location.href = '/login/email';
  };

  return (
    <div className="login-page">
      <div className="logo-container">
        <div className="logo">
          <svg width="63" height="63" viewBox="0 0 63 63" fill="none" xmlns="http://www.w3.org/2000/svg">
            <rect x="2.5" y="2.5" width="58" height="58" rx="7.5" stroke="black" stroke-width="5"/>
            <rect x="16.5" y="16.5" width="31" height="31" rx="2.5" fill="#EADDFF" stroke="black" stroke-width="5"/>
          </svg>
        </div>
        <h1>Altru AI</h1>
      </div>

      <div className="login-container">
        <button className="login-button apple-button" onClick={handleAppleLogin}>
          <svg width="15" height="18" viewBox="0 0 15 18" fill="none" xmlns="http://www.w3.org/2000/svg">
            <path d="M11.1113 0.885223C11.1868 1.87123 10.875 2.84784 10.2421 3.60764C9.93898 3.9846 9.55432 4.28787 9.11705 4.49461C8.67978 4.70136 8.2013 4.80619 7.71764 4.80121C7.68678 4.32705 7.74968 3.85146 7.90273 3.40163C8.05579 2.95179 8.29601 2.53653 8.60964 2.17959C9.25112 1.44893 10.1443 0.986785 11.1113 0.885223Z" fill="black"/>
            <path d="M12.8752 8.06162C12.5538 8.6278 12.3809 9.26608 12.3725 9.91705C12.3733 10.6493 12.5896 11.3651 12.9945 11.9752C13.3994 12.5854 13.9749 13.0628 14.6493 13.3481C14.3842 14.2094 13.9845 15.0235 13.4651 15.76C12.7675 16.8034 12.0361 17.8229 10.8753 17.8417C10.3232 17.8545 9.95056 17.6957 9.56224 17.5303C9.15719 17.3577 8.7351 17.1779 8.07461 17.1779C7.37413 17.1779 6.93314 17.3635 6.50783 17.5425C6.14029 17.6972 5.78445 17.8469 5.28296 17.8677C4.17745 17.9087 3.33259 16.7541 2.60968 15.7204C1.16479 13.6094 0.0397102 9.77138 1.54804 7.15988C1.89625 6.53416 2.40061 6.0093 3.01197 5.63645C3.62334 5.26361 4.32086 5.05548 5.03657 5.03236C5.66357 5.01946 6.26521 5.26112 6.79269 5.47299C7.19609 5.63502 7.55611 5.77963 7.85092 5.77963C8.11008 5.77963 8.46006 5.64073 8.86793 5.47886C9.51042 5.22386 10.2966 4.91185 11.0977 4.99594C11.7116 5.01515 12.3127 5.17726 12.853 5.46938C13.3934 5.7615 13.8581 6.17559 14.2105 6.67876C13.6559 7.01972 13.1965 7.49543 12.8752 8.06162Z" fill="black"/>
          </svg>
          <span>Continue with Apple</span>
        </button>
        <button className="login-button dark-button" onClick={handleGoogleLogin}>
          <svg width="18" height="18" viewBox="0 0 18 18" fill="none" xmlns="http://www.w3.org/2000/svg">
            <path fill-rule="evenodd" clip-rule="evenodd" d="M17.16 9.19323C17.16 8.5905 17.1059 8.01095 17.0055 7.45459H9V10.7425H13.5746C13.3775 11.8051 12.7786 12.7053 11.8784 13.308V15.4407H14.6255C16.2327 13.961 17.16 11.7819 17.16 9.19323Z" fill="white"/>
            <path fill-rule="evenodd" clip-rule="evenodd" d="M8.99997 17.5C11.295 17.5 13.2191 16.7389 14.6254 15.4407L11.8784 13.3079C11.1173 13.8179 10.1436 14.1193 8.99997 14.1193C6.78611 14.1193 4.91224 12.6241 4.24383 10.615H1.40405V12.8173C2.80269 15.5952 5.67724 17.5 8.99997 17.5Z" fill="white"/>
            <path fill-rule="evenodd" clip-rule="evenodd" d="M4.24387 10.6151C4.07387 10.1051 3.97728 9.56031 3.97728 9.00008C3.97728 8.43985 4.07387 7.89508 4.24387 7.38508V5.1828H1.40409C0.82841 6.3303 0.5 7.62849 0.5 9.00008C0.5 10.3717 0.82841 11.6699 1.40409 12.8174L4.24387 10.6151Z" fill="white"/>
            <path fill-rule="evenodd" clip-rule="evenodd" d="M8.99997 3.88075C10.2479 3.88075 11.3684 4.30961 12.2493 5.15189L14.6873 2.71393C13.2152 1.34234 11.2911 0.500061 8.99997 0.500061C5.67724 0.500061 2.80269 2.40484 1.40405 5.1828L4.24383 7.38507C4.91224 5.37598 6.78611 3.88075 8.99997 3.88075Z" fill="white"/>
          </svg>
          <span>Continue with Google</span>
        </button>
        <button className="login-button dark-button" onClick={handleMetaLogin}>
          <svg width="22" height="15" viewBox="0 0 22 15" fill="none" xmlns="http://www.w3.org/2000/svg">
            <path d="M11.2851 3.46585C9.61012 1.3332 8.20948 0.520508 6.53314 0.520508C3.11557 0.520508 0.49707 4.96812 0.49707 9.67564C0.49707 12.6215 1.92221 14.4795 4.30934 14.4795C6.02741 14.4795 7.26306 13.6696 9.45964 9.82975C9.45964 9.82975 10.3754 8.21274 11.0053 7.09882C11.226 7.45519 11.458 7.83864 11.7027 8.25085L12.7329 9.98372C14.7393 13.3416 15.8575 14.4795 17.8832 14.4795C20.2088 14.4795 21.5029 12.5961 21.5029 9.5891C21.5029 4.66004 18.8254 0.520508 15.5727 0.520508C13.8503 0.520508 12.504 1.81786 11.2851 3.46585ZM14.5062 8.4386L13.2746 6.38454C12.9414 5.84251 12.622 5.34431 12.3138 4.88771C13.4237 3.17439 14.3394 2.3208 15.4284 2.3208C17.6907 2.3208 19.5006 5.65169 19.5006 9.74307C19.5006 11.3026 18.9897 12.2075 17.9313 12.2075C16.9169 12.2075 16.4323 11.5376 14.5062 8.4386ZM2.76896 9.71417C2.76896 6.29659 4.47294 2.8021 6.50425 2.8021C7.60421 2.8021 8.52341 3.43738 9.93145 5.45305C8.59446 7.50377 7.78456 8.78995 7.78456 8.78995C6.00368 11.5817 5.38753 12.2075 4.39602 12.2075C3.37548 12.2075 2.76896 11.3116 2.76896 9.71417Z" fill="white"/>
          </svg>
          <span>Continue with Meta</span>
        </button>
        <button className="login-button dark-button" onClick={handleEmailLogin}>
          <svg width="20" height="20" viewBox="0 0 24 24">
            <path d="M20 4H4c-1.1 0-2 .9-2 2v12c0 1.1.9 2 2 2h16c1.1 0 2-.9 2-2V6c0-1.1-.9-2-2-2zm0 4l-8 5-8-5V6l8 5 8-5v2z" fill="currentColor"/>
          </svg>
          <span>Log in with e-mail</span>
        </button>
      </div>
    </div>
  )
}

export default LoginPage 